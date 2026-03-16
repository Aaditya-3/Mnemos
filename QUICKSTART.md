# Quick Start Guide

## Get Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env`
```bash
# Option A
python setup_env.py

# Option B
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac
```

Set at minimum for a full local run:
```env
GROQ_API_KEY=your_key_here
JWT_SECRET=replace_with_a_strong_secret
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=mnemos_semantic_memory
REQUIRE_QDRANT=true
```

### 3. Start Backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open App
Go to:
```text
http://localhost:8000
```

Optional production frontend scaffold:
```bash
cd frontend-vite
npm install
npm run dev
```

## Semantic Memory + Streaming Defaults

```env
ENABLE_SEMANTIC_MEMORY=true
ENABLE_STREAMING=true
ENABLE_TOOLS=true
```

Semantic vector backend:
```env
DATABASE_URL=sqlite:///memory/app.db
QDRANT_URL=https://your-qdrant-instance
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=mnemos_semantic_memory
REQUIRE_QDRANT=true
```

Notes:

- For a local Qdrant instance, use `QDRANT_URL=http://localhost:6333`.
- If you set `REQUIRE_QDRANT=false`, the app can still start without Qdrant, but semantic vector retrieval is disabled. There is no active JSON semantic fallback in the current code path.

## Useful Endpoints

- `POST /chat`
- `POST /chat/stream`
- `POST /chat/agent`
- `GET /tools`
- `GET /settings`
- `POST /settings`
- `GET /memories/semantic`
- `DELETE /memories/semantic/{memory_id}`
- `POST /admin/semantic/decay`
- `POST /admin/semantic/compress`
- `POST /admin/semantic/reembed`
- `GET /metrics`
- `GET /health`
- `POST /auth/token/issue`
- `POST /auth/token/rotate`

## Verification Checklist

- [ ] Backend starts cleanly and `GROQ_API_KEY loaded: True` is shown.
- [ ] `POST /chat` returns `usage` and `chat_id`.
- [ ] `POST /chat/stream` emits `start`, `token`, and `done` events.
- [ ] `GET /memories/semantic` returns semantic memory rows after chatting.
- [ ] `GET /metrics` exposes request/latency/token metrics.
