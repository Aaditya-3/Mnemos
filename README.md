# Mnemos: AI Chat with Long-Term Memory

Mnemos is a full-stack chat application built with FastAPI + React, powered by Groq for generation, with layered memory (deterministic, structured, and semantic).

This README reflects the current implementation in this repository.

## Current Project Level

Implemented and working now:

- Username/password auth (`/auth/signup`, `/auth/login`) with per-user chat isolation
- Multi-chat session management (`/chats`, `/chats/new`, `/chats/{chat_id}`)
- Main chat endpoint with orchestrator pipeline (`/chat`)
- SSE streaming chat endpoint (`/chat/stream`)
- Agent/tool route (`/chat/agent`) with structured tool execution
- Deterministic memory + structured "brain" memory + semantic vector memory
- Semantic memory maintenance routes (decay, compress, re-embed)
- Observability endpoints (`/metrics`, `/health`) and request/security middleware
- Built-in frontend served by backend at `/` (CDN React + Tailwind + Babel)
- Optional Vite frontend (`frontend-vite/`) for production-style frontend workflow

## Architecture (Current)

```text
Frontend (served by FastAPI at /)
-> API layer (FastAPI routes)
-> Brain layer (intent, structured extraction, deterministic responses)
-> Orchestrator (context build/rank/assemble)
-> Groq LLM (generation)
-> Persistence (SQLAlchemy relational DB + memory JSON/vector store)
-> Background hooks (semantic ingest/decay/compress/re-embed)
```

## Repository Structure

```text
.
|-- backend/
|   |-- main.py                          # FastAPI entrypoint
|   `-- app/
|       |-- api/platform.py              # runtime architecture/config introspection endpoints
|       |-- brain/                       # intent + structured memory layer
|       |-- config/                      # runtime/ranking/tool config
|       |-- core/                        # middleware/auth/db/llm/tool primitives
|       |-- embeddings/                  # embedding provider abstraction
|       |-- llm/                         # LLM client adapter and retries
|       |-- observability/               # logging + metrics
|       |-- orchestrator/                # prompt/context/stream pipeline
|       |-- security/                    # replay + JWT rotation utilities
|       |-- services/                    # semantic memory/tool/streaming services
|       |-- tasks/                       # background task entrypoints + celery wiring
|       `-- tools/                       # calculator/currency/web_search tools
|-- frontend/                            # default UI served at /
|-- frontend-vite/                       # optional Vite React frontend
|-- memory/                              # deterministic + semantic fallback stores
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create `.env`

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Minimum required:

```env
GROQ_API_KEY=your_key_here
```

Recommended before any shared deployment:

```env
JWT_SECRET=replace_with_a_strong_secret
APP_ENV=dev
```

### 3. Run backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open app

```text
http://localhost:8000
```

The backend serves `frontend/index.html` directly at `/`.

## Optional Frontend (Vite)

Use this only if you want a separate frontend dev/build workflow.

```bash
cd frontend-vite
npm install
npm run dev
```

Vite runs on `http://localhost:5173` and proxies API routes to `http://localhost:8000`.

## Auth and User Identity

The UI login/signup flow uses:

- `POST /auth/signup`
- `POST /auth/login`

Both return `user_id`. Use that value in `X-User-ID` for user-scoped routes.

Example:

```bash
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"password123"}'
```

Then call chat:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-User-ID: <returned_user_id>" \
  -d '{"message":"I prefer Python for backend work"}'
```

Optional token routes are also available:

- `POST /auth/token/issue` (requires `X-User-ID`)
- `POST /auth/token/rotate`

## API Endpoints (Current)

### Auth

- `POST /auth/signup`
- `POST /auth/login`
- `POST /auth/token/issue`
- `POST /auth/token/rotate`

### Chat

- `POST /chat`
- `POST /chat/stream`
- `POST /chat/agent`

### Chat Sessions

- `GET /chats`
- `GET /chats/{chat_id}`
- `DELETE /chats/{chat_id}`
- `POST /chats/new`

### User Settings

- `GET /settings`
- `POST /settings`

### Memory

- `GET /memories` (deterministic memory store view)
- `GET /memories/structured` (brain-layer structured memory)
- `GET /memories/semantic`
- `DELETE /memories/semantic/{memory_id}`

### Semantic Admin

- `POST /admin/semantic/decay`
- `POST /admin/semantic/compress`
- `POST /admin/semantic/reembed`

### Tools and Platform

- `GET /tools`
- `GET /api/v1/platform/architecture`
- `GET /api/v1/platform/config`

### Ops

- `GET /metrics`
- `GET /health`

### Frontend Assets

- `GET /` (serves `frontend/index.html`)
- `GET /app.jsx`
- `GET /script.js` (legacy file still exposed)

Note: most user-specific routes require `X-User-ID`.

## Streaming Events (`/chat/stream`)

Server-Sent Events emitted by stream handler:

- `start`
- `tool_call`
- `token`
- `done`
- `error`

## Memory System

Mnemos currently uses three memory layers:

1. Deterministic key/value memory
- Backed by `memory/memories.json` (or Mongo if enabled)
- Used for explicit profile/preferences retrieval

2. Structured brain memory
- Intent classification + structured extraction
- Stores preferences/events in relational tables (`user_preferences`, `user_events`)
- Handles direct logical answers for memory queries when possible

3. Semantic vector memory
- Ingests user messages into semantic memory nodes with embeddings
- Retrieval ranks by similarity + importance + recency
- Qdrant is preferred backend; local JSON fallback when Qdrant is unavailable

## Tooling

Registered tools:

- `calculator`
- `currency_convert`
- `web_search`

Tool calling is available through `/chat/agent` and optionally through `/chat` when `use_tools=true`.

## Data and Persistence

Relational DB (`DATABASE_URL`, default `sqlite:///memory/app.db`) stores:

- users
- chat sessions/messages
- usage logs
- user settings
- structured preferences/events

Semantic memory backend:

- Qdrant collection (`QDRANT_COLLECTION`) when Qdrant is reachable
- fallback file: `memory/semantic_memories.json`

## Key Environment Variables

See `.env.example` for the full list. Most important runtime knobs:

```env
# Required for generation
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

# Core runtime
APP_ENV=dev
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///memory/app.db

# Semantic memory + embeddings
ENABLE_SEMANTIC_MEMORY=true
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-small-en-v1.5
EMBEDDING_DIMS=384
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=mnemos_semantic_memory
REQUIRE_QDRANT=false

# Feature toggles
ENABLE_STREAMING=true
ENABLE_TOOLS=true
ENABLE_BACKGROUND_TASKS=true
ENABLE_REALTIME_WEB=true

# Security and limits
JWT_SECRET=CHANGE_ME_SECRET
RATE_LIMIT_RPM=60
ENABLE_PROMPT_GUARD=true
MAX_PROMPT_CHARS=6000
ENABLE_TOOL_SANDBOX=true
```

## Operational Notes

- If backend startup prints `GROQ_API_KEY loaded: False`, `.env` is missing/misconfigured.
- `/health` reports active vector backend and feature toggle states.
- `/metrics` exposes Prometheus-style counters/histograms.
- In-memory rate limits/replay tracking and refresh-token store are suitable for local/single-instance use; move these to shared infrastructure (Redis/DB) for multi-instance deployments.

## Current Limits and Tradeoffs

- LLM generation is Groq-only in current code path.
- Local JSON vector fallback is convenient for development but not for distributed consistency.
- Optional Celery wiring exists, but FastAPI `BackgroundTasks` is the default execution path.
- Legacy frontend file `frontend/script.js` is still present; primary UI is `frontend/app.jsx`.
