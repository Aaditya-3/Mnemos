# AI Chat System with Long-Term Memory

A complete chat application powered by Groq with persistent memory capabilities. The system remembers user preferences, constraints, and facts across conversations.

## 🏗️ Architecture

```
Frontend (HTML + JS)
        ↓
FastAPI Backend
        ↓
Memory Extraction (optional)
        ↓
Memory Store (persistent)
        ↓
Memory Retrieval
        ↓
Groq LLM
        ↓
Response back to user
```

## 📁 Project Structure

```
project_root/
│
├── backend/
│   └── app.py                 # FastAPI server
│
├── llm/
│   └── app/core/llm/groq_client.py   # Groq LLM client
│
├── memory/
│   ├── memory_schema.py       # Memory data structure
│   ├── memory_store.py        # In-memory storage
│   ├── memory_extractor.py    # Extract memories from messages
│   └── memory_retriever.py    # Retrieve relevant memories
│
├── frontend/
│   ├── index.html             # Chat UI
│   └── script.js              # Frontend logic
│
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

**CRITICAL:** You must create a `.env` file before starting the backend.

1. Copy `.env.example` to `.env`:
   ```bash
   # Windows
   copy .env.example .env
   
   # Linux/Mac
   cp .env.example .env
   ```

2. The `.env.example` already contains an example API key. For production, replace it with your own:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```

   Get your API key from: https://console.groq.com

3. **Verify the `.env` file exists** in the project root (same directory as `requirements.txt`)

### 3. Run Backend

**Backend MUST be started from project root.** The `.env` file is loaded from the current working directory at startup.

From the project root directory (where `requirements.txt` and `.env` are located):

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Or with auto-reload:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start on `http://localhost:8000`

**Verify Setup:**
When the backend starts, you must see:
```
GROQ_API_KEY loaded: True
```
If you see `GROQ_API_KEY loaded: False`, the `.env` file was not found (wrong directory or missing file).

**If you see `GROQ_API_KEY loaded: False`:**
1. **Create the `.env` file** (if you haven't already):
   ```bash
   python setup_env.py
   # OR manually: copy .env.example .env
   ```
2. **Restart the backend** - The `.env` file is only loaded at startup
3. Verify the `.env` file exists in the project root (same folder as `requirements.txt`)
4. Check the file contains `GROQ_API_KEY=...` (no spaces around `=`)
5. Test environment loading: `python test_env.py`

### 4. Open Frontend

The backend now serves the frontend automatically! Simply open your browser and go to:

```
http://localhost:8000
```

The frontend will be served from the root URL, and all API calls will work automatically.

**Note:** If you prefer to serve the frontend separately, you can still open `frontend/index.html` directly in your browser, but you'll need to update the `API_URL` in `script.js` back to `http://localhost:8000/chat`.

## 💾 Memory System

### What Gets Stored

The system extracts and stores **long-term, reusable information**:

- ✅ **Preferences**: Language, style, format preferences
- ✅ **Constraints**: Time limits, budget, availability
- ✅ **Stable Facts**: Name, location, role, skills

### What Doesn't Get Stored

- ❌ Greetings (hello, hi, thanks)
- ❌ Emotions (happy, sad, excited)
- ❌ One-off messages
- ❌ Temporary states

### Memory Confidence

- **Initial confidence**: 0.7
- **Confirmation** (same info repeated): +0.1
- **Contradiction** (different value): -0.3
- **Deletion**: Memories below 0.3 confidence are removed

## 🎯 Example Demo Conversation

```
User: Hi, I'm John and I prefer Python for coding.

Assistant: Hello John! I'll remember that you prefer Python for coding.

User: I'm working on a web project with a tight deadline.

Assistant: I understand you're working on a web project with a tight deadline. 
            I'll keep that in mind. Since you prefer Python, are you using 
            frameworks like Flask or FastAPI?

User: What's my preferred programming language?

Assistant: Your preferred programming language is Python.
```

## 🔒 Security

- ✅ API keys stored in `.env` (never committed)
- ✅ No API keys in frontend code
- ✅ Backend handles all API calls
- ✅ CORS enabled for development

## 🛠️ API Endpoints

### POST `/chat`

Send a message to the AI.

**Request:**
```json
{
  "message": "Your message here"
}
```

**Response:**
```json
{
  "reply": "AI response here"
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "api_key_loaded": true
}
```

## 📝 Notes

- Memory is stored in-memory (single user, single session)
- For production, consider persistent storage (database)
- Backend never crashes - all errors returned as JSON
- Uses Groq `llama-3.1-8b-instant` model for fast responses

## 🐛 Troubleshooting

### GROQ_API_KEY loaded: False

**This is the most common issue. Fix it by:**

1. **Verify `.env` file exists:**
   ```bash
   # From project root, check if .env exists
   dir .env        # Windows
   ls -la .env     # Linux/Mac
   ```

2. **Check `.env` file format:**
   - Must be in project root (same folder as `requirements.txt`)
   - Must contain exactly: `GROQ_API_KEY=your_key_here`
   - No spaces around `=`
   - No quotes around the key
   - No trailing spaces

3. **Verify you're running from project root:**
   ```bash
   # Make sure you're in the directory with requirements.txt
   pwd              # Linux/Mac
   cd               # Windows (shows current directory)
   ```

4. **Restart the backend** after creating/editing `.env`

### Backend won't start:
- Check that `GROQ_API_KEY` is set in `.env`
- Verify Python dependencies are installed: `pip install -r requirements.txt`
- Check port 8000 is not in use
- Ensure you're running from project root directory

### Frontend can't connect:
- Ensure backend is running on `http://localhost:8000`
- Check browser console (F12) for CORS errors
- Update `API_URL` in `script.js` if backend is on different port
- Try accessing `http://localhost:8000/health` directly in browser

### No memory being stored:
- Memory extraction is automatic and optional
- Try explicit statements like "I prefer X" or "My name is Y"
- Check backend logs for extraction errors (non-fatal errors are logged but don't crash)
- Memory requires confidence > 0.5 to be retrieved

### Groq API errors:
- Verify your API key is valid
- Check your internet connection
- Ensure you have API quota remaining
- Check backend logs for specific error messages
