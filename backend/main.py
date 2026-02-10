"""
FastAPI Backend

Main API server for the chat application.
Uses Groq as the only LLM provider.
"""

# CRITICAL: Load .env FIRST, before any other imports that use env vars.
from pathlib import Path
from dotenv import load_dotenv

# Load from project root by path (works regardless of current working directory)
_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root / ".env"
load_dotenv(dotenv_path=_env_file)
if _env_file.exists():
    print(f"Loaded .env from: {_env_file}")
else:
    load_dotenv()  # fallback: current directory

import os
print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))

from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid

from memory.memory_extractor import extract_memory
from memory.memory_retriever import retrieve_memories
from backend.app.core.llm.groq_client import generate_response

project_root = _project_root
api_key_loaded = bool(os.getenv("GROQ_API_KEY"))
if not api_key_loaded:
    print("WARNING: GROQ_API_KEY not found. Create .env in project root and set GROQ_API_KEY=...")

app = FastAPI(title="AI Chat with Memory")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = project_root / "frontend"
if frontend_path.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_path)), name="frontend")


class ChatSession:
    def __init__(self):
        self.id: str = str(uuid.uuid4())
        self.title: str = "New Chat"
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()
        self.messages: List[dict] = []


chat_sessions: dict[str, ChatSession] = {}
current_chat_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    chat_id: str


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    1. Receive user message
    2. Extract memory if present
    3. Retrieve relevant memories
    4. Build single combined prompt (memory + user message)
    5. Call Groq generate_response(prompt)
    6. Store messages and return response
    """
    try:
        user_message = request.message.strip()
        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Message cannot be empty"})

        chat_id = request.chat_id
        if not chat_id or chat_id not in chat_sessions:
            chat_session = ChatSession()
            words = user_message.split()[:5]
            chat_session.title = " ".join(words) + ("..." if len(user_message.split()) > 5 else "")
            chat_sessions[chat_session.id] = chat_session
            chat_id = chat_session.id
        else:
            chat_session = chat_sessions[chat_id]
            chat_session.updated_at = datetime.now()

        try:
            memory_extracted = extract_memory(user_message)
            if memory_extracted:
                print(f"✓ Memory extracted and stored: {memory_extracted.key} = {memory_extracted.value}")
            else:
                print("ℹ No memory extracted from this message")
        except Exception as e:
            print(f"⚠ Memory extraction error (non-fatal): {e}")

        memory_context = ""
        try:
            memory_context = retrieve_memories(user_message)
            if memory_context:
                print(f"✓ Using {len(memory_context.split(chr(10)))} memory items in context")
            else:
                print("ℹ No memories to include in context")
        except Exception as e:
            print(f"⚠ Memory retrieval error (non-fatal): {e}")
            memory_context = ""

        # Build ONE combined prompt and call Groq
        full_prompt = user_message
        if memory_context:
            full_prompt = f"""IMPORTANT CONTEXT ABOUT THE USER (remember and use this information):
{memory_context}

Now respond to the user's message below. Use the context above to personalize your response and remember who you're talking to.

User: {user_message}"""

        try:
            reply = generate_response(full_prompt)
        except RuntimeError as e:
            return JSONResponse(status_code=500, content={"error": f"AI service error: {str(e)}"})

        chat_session.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        chat_session.messages.append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().isoformat()
        })

        return ChatResponse(reply=reply, chat_id=chat_id)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in /chat: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal error: {str(e)}"})


@app.get("/")
async def root():
    index_path = project_root / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "AI Chat with Memory API",
        "status": "running",
        "endpoints": {"chat": "POST /chat", "health": "GET /health"},
        "api_key_loaded": api_key_loaded
    }


@app.get("/script.js")
async def serve_script():
    script_path = project_root / "frontend" / "script.js"
    if script_path.exists():
        return FileResponse(str(script_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found")


@app.get("/app.jsx")
async def serve_app():
    app_path = project_root / "frontend" / "app.jsx"
    if app_path.exists():
        return FileResponse(str(app_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.jsx not found")


@app.get("/chats", response_model=List[ChatSessionResponse])
async def get_chats():
    sessions = []
    for chat_id, session in sorted(chat_sessions.items(), key=lambda x: x[1].updated_at, reverse=True):
        sessions.append(ChatSessionResponse(
            id=session.id,
            title=session.title,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            message_count=len(session.messages)
        ))
    return sessions


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat not found")
    session = chat_sessions[chat_id]
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "messages": session.messages
    }


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat not found")
    del chat_sessions[chat_id]
    return {"status": "deleted", "chat_id": chat_id}


@app.post("/chats/new")
async def create_new_chat():
    chat_session = ChatSession()
    chat_sessions[chat_session.id] = chat_session
    return ChatSessionResponse(
        id=chat_session.id,
        title=chat_session.title,
        created_at=chat_session.created_at.isoformat(),
        updated_at=chat_session.updated_at.isoformat(),
        message_count=0
    )


@app.get("/memories")
async def get_memories():
    from memory.memory_store import get_memory_store
    store = get_memory_store()
    memories = store.get_all_memories()
    return {
        "total": len(memories),
        "memories": [
            {
                "id": m.id,
                "type": m.type,
                "key": m.key,
                "value": m.value,
                "confidence": m.confidence,
                "created_at": m.created_at.isoformat(),
                "last_updated": m.last_updated.isoformat()
            }
            for m in memories
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "api_key_loaded": api_key_loaded}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
