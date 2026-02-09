# How to Run the Application

## 🚀 Single Command Setup

**Everything runs from ONE command!** The backend serves the frontend automatically.

### Step 1: Navigate to Project Root
```bash
cd c:\temp2
```

### Step 2: Start the Server
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Or using Python directly:
```bash
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: Open Browser
Open your browser and go to:
```
http://localhost:8000
```

## ✅ That's It!

- ✅ **Backend API** runs on `http://localhost:8000`
- ✅ **Frontend React App** is served at `http://localhost:8000`
- ✅ **No separate frontend server needed**
- ✅ **Everything works together automatically**

## 📋 What Happens

1. Backend starts and loads `.env` file
2. FastAPI serves the React frontend from `/`
3. Frontend makes API calls to `/chat`, `/chats`, etc.
4. Everything runs on port 8000

## 🔄 Development Mode

The `--reload` flag means:
- Backend auto-reloads when you change Python files
- Frontend files are served directly (no build step needed)
- Just refresh your browser to see frontend changes

## 🛑 To Stop

Press `Ctrl+C` in the terminal where the server is running.

## ❓ FAQ

**Q: Do I need to run frontend separately?**  
A: No! The backend serves the frontend automatically.

**Q: Do I need npm or node?**  
A: No! We're using React via CDN, so no build step needed.

**Q: Can I run on a different port?**  
A: Yes, change `--port 8000` to your desired port, then update the frontend API_URL if needed.

**Q: What if I see errors?**  
A: Make sure:
- You're in the project root directory (`c:\temp2`)
- `.env` file exists with `GEMINI_API_KEY=...`
- Port 8000 is not already in use
- All dependencies are installed: `pip install -r requirements.txt`
