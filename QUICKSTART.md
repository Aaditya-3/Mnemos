# Quick Start Guide

## 🚀 Get Running in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Create .env File
```bash
# Option A: Use the helper script
python setup_env.py

# Option B: Manual copy
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac
```

**Important:** The `.env.example` already contains an API key. For production, replace it with your own key from https://makersuite.google.com/app/apikey

### Step 3: Start Backend (Serves Frontend Too!)
```bash
# From project root directory
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

**Verify:** You should see:
```
Loaded .env from: C:\temp2\.env
GEMINI_API_KEY loaded: True
API key preview: AIzaSyCmnn...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 4: Open Frontend
**Just open your browser and go to:**
```
http://localhost:8000
```

That's it! The backend serves both the API and the frontend React app.

## ✅ Verification Checklist

- [ ] `.env` file exists in project root
- [ ] Backend shows `GEMINI_API_KEY loaded: True`
- [ ] Backend is running on `http://localhost:8000`
- [ ] Frontend can connect to backend
- [ ] Test message gets a response from Gemini

## 🐛 If GEMINI_API_KEY loaded: False

1. Check `.env` file exists: `dir .env` (Windows) or `ls -la .env` (Linux/Mac)
2. Verify file format: `GEMINI_API_KEY=your_key` (no spaces around `=`)
3. Ensure you're running from project root (where `requirements.txt` is)
4. Restart backend after creating/editing `.env`

## 📝 Test the Memory System

Try these messages to test memory:

1. "My name is Alice and I prefer Python programming"
2. "What's my preferred programming language?"
3. "I'm working on a project with a deadline next week"
4. "What did I tell you about my deadline?"

The system should remember your preferences and facts across messages!
