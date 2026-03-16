# Mnemos

Mnemos is a full-stack AI chat application built with FastAPI and React. It uses Groq for generation and combines three memory layers: deterministic profile memory, structured "brain" memory, and semantic vector memory in Qdrant.

This README reflects the current final state of the project in this repository.

## Final Project Snapshot

- Username/password auth with per-user isolation
- Multi-chat sessions with persisted history
- Standard chat, streaming chat, and tool-enabled agent chat endpoints
- Deterministic memory, structured relational memory, and semantic vector memory
- Semantic maintenance flows for decay, compression, and re-embedding
- Built-in frontend served by FastAPI at `/`
- Optional Vite frontend for a separate frontend workflow
- Health, metrics, and runtime inspection endpoints

## Architecture

```text
Frontend (served by FastAPI at /)
-> FastAPI routes
-> Brain layer
   - intent detection
   - structured memory extraction
   - deterministic memory lookup
   - temporal reasoning
-> Orchestrator
   - context build
   - semantic retrieval
   - ranking and prompt assembly
-> Groq LLM
-> Persistence
   - SQLite relational database
   - memory/memories.json deterministic store
   - Qdrant semantic vector store
-> Background semantic maintenance
   - ingest
   - decay
   - compression
   - re-embedding
```

## Tech Stack

- Backend: FastAPI, Uvicorn, SQLAlchemy
- LLM: Groq
- Embeddings: local sentence-transformers provider by default
- Vector store: Qdrant with `qdrant-client==1.16.2`
- Relational storage: SQLite by default
- Frontend: React with a built-in static UI, plus an optional Vite app
- Observability: Prometheus metrics and structured runtime logging

## Repository Structure

```text
.
|-- backend/
|   |-- main.py                          # FastAPI entrypoint
|   `-- app/
|       |-- api/platform.py              # runtime architecture/config endpoints
|       |-- brain/                       # structured memory + intent/temporal reasoning
|       |-- config/                      # runtime and ranking config
|       |-- core/                        # auth, db, llm, middleware primitives
|       |-- embeddings/                  # embedding provider abstraction
|       |-- observability/               # logging and metrics
|       |-- orchestrator/                # context build/rank/assemble pipeline
|       |-- services/                    # semantic memory, streaming, tool routing
|       |-- tasks/                       # background task entrypoints
|       `-- tools/                       # calculator, currency, web search
|-- frontend/                            # built-in UI served at /
|-- frontend-vite/                       # optional Vite frontend
|-- memory/                              # deterministic store, Qdrant service, legacy data files
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Setup

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

`backend/main.py` loads `.env` from the project root before the rest of the app starts.

Minimum values to set:

```env
GROQ_API_KEY=your_groq_api_key
JWT_SECRET=replace_with_a_strong_secret
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=mnemos_semantic_memory
REQUIRE_QDRANT=true
```

Notes:

- For local Qdrant, use `QDRANT_URL=http://localhost:6333`.
- If you set `REQUIRE_QDRANT=false`, the app can still boot when Qdrant is unreachable, but semantic vector retrieval will be disabled. The current code does not fall back to a JSON semantic vector store.
- The default embedding setup uses the local `bge-small-en-v1.5` sentence-transformers path.

### 3. Run the backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the app

```text
http://localhost:8000
```

The backend serves `frontend/index.html` directly at `/`.

## Optional Vite Frontend

Use this only if you want a separate frontend dev/build workflow.

```bash
cd frontend-vite
npm install
npm run dev
```

Vite runs on `http://localhost:5173` and proxies API calls to `http://localhost:8000`.

## Auth and First Request

The mounted auth flow is the simple username/password router:

- `POST /auth/signup`
- `POST /auth/login`

Both return a `user_id`. Use that value in `X-User-ID` for user-scoped endpoints.

Example signup:

```bash
curl -X POST http://localhost:8000/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"password123"}'
```

Example chat call:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-User-ID: <returned_user_id>" \
  -d '{"message":"I prefer Python for backend work"}'
```

Token helpers are also available:

- `POST /auth/token/issue`
- `POST /auth/token/rotate`

## API Surface

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

### Settings

- `GET /settings`
- `POST /settings`

### Memory

- `GET /memories`
- `GET /memories/structured`
- `GET /memories/semantic`
- `DELETE /memories/semantic/{memory_id}`
- `POST /test-memory` - semantic-memory smoke-test endpoint

### Semantic Maintenance

- `POST /admin/semantic/decay`
- `POST /admin/semantic/compress`
- `POST /admin/semantic/reembed`

### Tools and Runtime Introspection

- `GET /tools`
- `GET /api/v1/platform/architecture`
- `GET /api/v1/platform/config`

### Operations

- `GET /metrics`
- `GET /health`

### Frontend Assets

- `GET /`
- `GET /app.jsx`
- `GET /script.js`

Most user-specific routes require `X-User-ID`.

## Streaming Events

`/chat/stream` emits Server-Sent Events with these event types:

- `start`
- `tool_call`
- `token`
- `done`
- `error`

## Memory Model

Mnemos uses three memory layers.

### 1. Deterministic memory

- Stored in `memory/memories.json`
- Used for explicit profile and preference style retrieval
- Managed by the legacy deterministic memory extractor/store path

### 2. Structured brain memory

- Stored in relational tables such as `user_preferences` and `user_events`
- Supports intent-aware and logic-aware answers for profile and date queries
- Serves as the main structured source of truth for extracted user state

### 3. Semantic vector memory

- Embeds user messages into semantic memory nodes
- Stores semantic points in the Qdrant collection named by `QDRANT_COLLECTION`
- Retrieves by similarity, then reranks by similarity, importance, and recency
- Supports maintenance flows for decay, compression, deletion, and re-embedding

Important: the current semantic path is Qdrant-backed. There is no active JSON semantic vector fallback in the current repository code path.

## Data and Persistence

Current source-of-truth storage:

- `memory/app.db`
  - users
  - chat sessions and messages
  - usage logs
  - user settings
  - structured preferences
  - structured events
- `memory/memories.json`
  - deterministic key/value memory
- Qdrant collection `QDRANT_COLLECTION`
  - semantic vector memory points

Legacy JSON files may still exist in `memory/` from earlier iterations, including `semantic_memories.json`, `structured_memory.json`, `chat_sessions.json`, and `users.json`. The current runtime source of truth is SQLite plus Qdrant plus `memory/memories.json`.

## Tools

Registered tools:

- `calculator`
- `currency_convert`
- `web_search`

Tool execution is available through `/chat/agent` and can also be enabled through the main chat flow when tool routing is active.

## Key Environment Variables

See `.env.example` for the full template. The most important runtime knobs are:

```env
# LLM
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

# Core runtime
APP_ENV=dev
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///memory/app.db

# Semantic memory
ENABLE_SEMANTIC_MEMORY=true
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-small-en-v1.5
EMBEDDING_DIMS=384
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=mnemos_semantic_memory
REQUIRE_QDRANT=true

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

## Operational Checks

- Startup should print that `.env` was loaded from the project root.
- `GET /health` shows whether semantic memory is enabled and which vector backend is active.
- `GET /metrics` exposes Prometheus-style metrics.
- `GET /api/v1/platform/config` shows the active vector collection name and runtime toggles.
- `POST /test-memory` is useful when you want a quick semantic memory ingestion/retrieval smoke test.

## Current Tradeoffs

- LLM generation is Groq-only in the current runtime path.
- The mounted auth flow is a simple local username/password system.
- FastAPI `BackgroundTasks` is the default semantic maintenance execution path; Celery wiring exists but is optional.
- `frontend/script.js` is still exposed for compatibility, but `frontend/app.jsx` is the primary frontend entrypoint.
