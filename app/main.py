"""
FIN-1838 / QIDP-255 — OpenBB Workspace chat agent for QID stack.

Endpoints:
  GET  /agents.json   — discovery doc for pro.openbb.co Workspace
  POST /v1/query      — SSE chat stream (OpenBB Workspace protocol)
  GET  /health        — liveness probe

Capabilities:
  1. crucix_ideas       — fetch graphiti qid_intelligence episodes, summarize via Claude
  2. tradefarm_status   — query tradefarm FastAPI for agent equity / positions / trades
"""
from __future__ import annotations

import json
import os
from typing import AsyncGenerator, Optional

import anthropic
import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────

API_BEARER_TOKEN = os.environ.get("API_BEARER_TOKEN", "")
GRAPHITI_BASE_URL = os.environ.get(
    "GRAPHITI_BASE_URL", "http://graphiti-mcp.qid.svc.cluster.local:8000"
)
GRAPHITI_GROUP_ID = os.environ.get("GRAPHITI_GROUP_ID", "qid_intelligence")
TRADEFARM_URL = os.environ.get(
    "TRADEFARM_URL", "http://tradefarm-backend.tradefarm.svc.cluster.local:8000"
)
# Azure Anthropic proxy — sk-proxy is the in-cluster no-auth sentinel key.
# Routes to the team's Azure Anthropic subscription; no separate Anthropic key needed.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "sk-proxy")
ANTHROPIC_BASE_URL = os.environ.get(
    "ANTHROPIC_BASE_URL",
    "http://azure-anthropic-proxy.cloudtorch.svc.cluster.local:4000/v1",
)
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
# Max graphiti episodes to fetch per request (cost vs. coverage tradeoff)
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", "200"))

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="QID OpenBB Chat Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co", "http://localhost:*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Auth ───────────────────────────────────────────────────────────────────────


async def verify_bearer(authorization: str = Header(...)) -> None:
    scheme, _, token = authorization.partition(" ")
    if not API_BEARER_TOKEN:
        return  # dev: no token set = open
    if scheme.lower() != "bearer" or token != API_BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Discovery ──────────────────────────────────────────────────────────────────


@app.get("/agents.json")
async def agents_json() -> dict:
    return {
        "agents": [
            {
                "name": "QID Intelligence",
                "description": (
                    "Chat with your trading stack. Ask about crucix macro/equity trade ideas "
                    "or tradefarm paper-trading agent status."
                ),
                "capabilities": [
                    {
                        "name": "crucix_ideas",
                        "description": "Get the latest crucix trade ideas for a ticker or sector",
                        "examples": [
                            "What does crucix think about gold?",
                            "Show me recent crucix ideas on energy ETFs",
                            "Any crucix ideas on silver this week?",
                        ],
                    },
                    {
                        "name": "tradefarm_status",
                        "description": "Check tradefarm paper-trading agent equity, positions, and trades",
                        "examples": [
                            "How is agent 43 doing?",
                            "Show me the top 5 agents by equity",
                            "Which agents have open positions?",
                        ],
                    },
                ],
            }
        ]
    }


# ── Request / Response models ──────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    messages: list[ChatMessage]
    context: Optional[dict] = None


# ── Data fetchers ──────────────────────────────────────────────────────────────


async def _fetch_crucix_episodes() -> list[dict]:
    """Fetch recent crucix_idea_* episodes from graphiti REST API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                f"{GRAPHITI_BASE_URL}/v1/episodes",
                params={"group_id": GRAPHITI_GROUP_ID, "last_n": MAX_EPISODES},
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            episodes: list[dict] = data if isinstance(data, list) else data.get("episodes", [])
            # Keep only crucix_idea_* episodes (skip sweep narratives, outcomes, etc.)
            return [
                e for e in episodes
                if str(e.get("name", "")).startswith("crucix_idea_")
            ]
        except Exception:
            return []


async def _fetch_tradefarm_agents() -> list[dict]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{TRADEFARM_URL}/agents")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []


async def _fetch_tradefarm_trades(agent_id: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{TRADEFARM_URL}/agents/{agent_id}/trades")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return []


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a trading intelligence assistant for QID / FinTorch, a quantitative trading firm.
You have access to two live data sources injected below.

Rules:
- Be concise. Cite specific tickers, directions, and P&L numbers from the data.
- Do not invent data. If the data doesn't answer the question, say so.
- For crucix ideas: summarize each relevant idea with ticker, direction (LONG/SHORT), \
one-sentence thesis, and idea date.
- For tradefarm: report equity, realized/unrealized P&L, open positions, and strategy name.
- Dates are UTC."""


# ── Streaming core ─────────────────────────────────────────────────────────────


async def _stream(user_message: str) -> AsyncGenerator[str, None]:
    lower = user_message.lower()

    # Route to relevant data sources
    want_crucix = any(
        w in lower
        for w in ["crucix", "idea", "think about", "saying about", "view on", "thesis", "long", "short"]
    )
    want_tradefarm = any(
        w in lower
        for w in ["agent", "tradefarm", "position", "equity", "pnl", "trade", "strategy"]
    )
    if not want_crucix and not want_tradefarm:
        want_crucix = True
        want_tradefarm = True

    sections: list[str] = []

    if want_crucix:
        episodes = await _fetch_crucix_episodes()
        if episodes:
            lines: list[str] = []
            for ep in episodes[:60]:
                name = ep.get("name", "?")
                content = str(ep.get("content", ep.get("source_description", "")))
                created = str(ep.get("created_at", ep.get("valid_at", "")))[:10]
                if content:
                    lines.append(f"[{name} | {created}] {content[:500]}")
            sections.append("## Crucix Trade Ideas (recent)\n" + "\n---\n".join(lines))
        else:
            sections.append("## Crucix Trade Ideas\n(graphiti unreachable — no data)")

    if want_tradefarm:
        agents = await _fetch_tradefarm_agents()
        if agents:
            # Check if the user is asking about a specific agent by number
            agent_ids_mentioned = [
                int(tok.lstrip("#"))
                for tok in lower.replace("-", " ").split()
                if tok.lstrip("#").isdigit() and len(tok.lstrip("#")) <= 4
            ]
            if agent_ids_mentioned:
                aid = agent_ids_mentioned[0]
                matched = [a for a in agents if a.get("id") == aid]
                trades = await _fetch_tradefarm_trades(aid)
                sections.append(
                    f"## Tradefarm Agent {aid}\n"
                    + json.dumps(matched[0] if matched else {"error": "not found"}, indent=2)
                    + f"\n\nRecent trades (up to 10):\n"
                    + json.dumps(trades[:10], indent=2)
                )
            else:
                top = sorted(agents, key=lambda a: a.get("equity", 0), reverse=True)[:10]
                sections.append(
                    "## Tradefarm Agents (top 10 by equity)\n"
                    + json.dumps(top, indent=2)
                )
        else:
            sections.append("## Tradefarm Agents\n(tradefarm API unreachable — no data)")

    system = _SYSTEM
    if sections:
        system += "\n\n# Live Data\n\n" + "\n\n".join(sections)

    client = anthropic.AsyncAnthropic(
        api_key=ANTHROPIC_API_KEY,
        base_url=ANTHROPIC_BASE_URL,
    )

    try:
        async with client.messages.stream(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield f"data: {json.dumps({'delta': text})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'delta': f'[LLM error: {exc}]'})}\n\n"

    yield "data: [DONE]\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.post("/v1/query")
async def query(body: QueryRequest, _: None = Depends(verify_bearer)) -> StreamingResponse:
    user_message = next(
        (m.content for m in reversed(body.messages) if m.role == "user"), ""
    )
    return StreamingResponse(
        _stream(user_message),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
