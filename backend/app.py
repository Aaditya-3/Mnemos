"""
FastAPI Backend

Main API server for the chat application.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid

from memory.memory_extractor import extract_memory
from memory.memory_retriever import retrieve_memories
from llm.gemini_client import call_gemini

# Load environment variables from project root
# Try multiple paths to ensure we find .env file
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

# Try loading from multiple locations
loaded = False

# 1. Try project root (where backend/ folder is)
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    loaded = True
    print(f"Loaded .env from: {env_path}")

# 2. Try current working directory
if not loaded:
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(dotenv_path=cwd_env, override=True)
        loaded = True
        print(f"Loaded .env from: {cwd_env}")

# 3. Fallback: try current directory (no path specified)
if not loaded:
    load_dotenv(override=True)
    # Check if it worked
    if os.getenv("GEMINI_API_KEY"):
        loaded = True
        print("Loaded .env from current directory")

# 4. Final fallback: try without override
if not loaded:
    load_dotenv(override=False)

# Check API key on startup
api_key = os.getenv("GEMINI_API_KEY")
api_key_loaded = bool(api_key)
print(f"GEMINI_API_KEY loaded: {api_key_loaded}")
if api_key_loaded:
    print(f"API key preview: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
else:
    print("WARNING: GEMINI_API_KEY not found. Create .env file from .env.example")

# Initialize FastAPI app
app = FastAPI(title="AI Chat with Memory")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend) - serves CSS, JS, etc.
frontend_path = project_root / "frontend"
if frontend_path.exists():
    # Mount frontend directory to serve static assets
    app.mount("/frontend", StaticFiles(directory=str(frontend_path)), name="frontend")


# Chat history storage (in-memory, single user)
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
    """Request model for chat endpoint."""
    message: str
    chat_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    reply: str
    chat_id: str


class ChatSessionResponse(BaseModel):
    """Response model for chat session."""
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Flow:
    1. Receive user message
    2. Extract memory if present
    3. Retrieve relevant memories
    4. Build prompt with memory context
    5. Call Gemini
    6. Store messages in chat session
    7. Return response
    """
    try:
        user_message = request.message.strip()
        
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )
        
        # Get or create chat session
        chat_id = request.chat_id
        if not chat_id or chat_id not in chat_sessions:
            chat_session = ChatSession()
            # Use first few words of message as title
            words = user_message.split()[:5]
            chat_session.title = " ".join(words) + ("..." if len(user_message.split()) > 5 else "")
            chat_sessions[chat_session.id] = chat_session
            chat_id = chat_session.id
        else:
            chat_session = chat_sessions[chat_id]
            chat_session.updated_at = datetime.now()
        
        # Step 1: Extract memory from user message (CRITICAL - must work)
        memory_extracted = None
        try:
            memory_extracted = extract_memory(user_message)
            if memory_extracted:
                print(f"✓ Memory extracted and stored: {memory_extracted.key} = {memory_extracted.value}")
            else:
                print("ℹ No memory extracted from this message")
        except Exception as e:
            # Log error but continue - memory extraction shouldn't block chat
            print(f"⚠ Memory extraction error (non-fatal): {e}")
        
        # Step 2: Retrieve relevant memories (ALWAYS include if available)
        memory_context = ""
        try:
            memory_context = retrieve_memories(user_message)
            if memory_context:
                print(f"✓ Using {len(memory_context.split(chr(10)))} memory items in context")
            else:
                print("ℹ No memories to include in context")
        except Exception as e:
            # Log error but continue
            print(f"⚠ Memory retrieval error (non-fatal): {e}")
            memory_context = ""
        
        # Step 3: Call Gemini with memory context merged into prompt
        try:
            reply = call_gemini(
                prompt=user_message,
                memory_context=memory_context
            )
        except RuntimeError as e:
            # LLM or API errors
            return JSONResponse(
                status_code=500,
                content={"error": f"AI service error: {str(e)}"}
            )
        
        # Step 4: Store messages in chat session
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
        
        # Step 5: Return response
        return ChatResponse(reply=reply, chat_id=chat_id)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Unexpected errors - backend must never crash
        print(f"Unexpected error in /chat: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal error: {str(e)}"}
        )


@app.get("/")
async def root():
    """Serve the frontend index.html."""
    index_path = project_root / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        return {
            "message": "AI Chat with Memory API",
            "status": "running",
            "endpoints": {
                "chat": "POST /chat",
                "health": "GET /health"
            },
            "api_key_loaded": api_key_loaded
        }


@app.get("/script.js")
async def serve_script():
    """Serve the frontend script.js."""
    script_path = project_root / "frontend" / "script.js"
    if script_path.exists():
        return FileResponse(str(script_path), media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="script.js not found")


@app.get("/app.jsx")
async def serve_app():
    """Serve the frontend app.jsx."""
    app_path = project_root / "frontend" / "app.jsx"
    if app_path.exists():
        return FileResponse(str(app_path), media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="app.jsx not found")


@app.get("/chats", response_model=List[ChatSessionResponse])
async def get_chats():
    """Get all chat sessions."""
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
    """Get a specific chat session with messages."""
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
    """Delete a chat session."""
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat not found")
    del chat_sessions[chat_id]
    return {"status": "deleted", "chat_id": chat_id}


@app.post("/chats/new")
async def create_new_chat():
    """Create a new chat session."""
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
    """Get all stored memories (for debugging)."""
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
    """Health check endpoint."""
    return {"status": "ok", "api_key_loaded": api_key_loaded}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
