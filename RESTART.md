# Restart Instructions (after model/cache changes)

## 1. Clean Python bytecode cache (optional but recommended)

From **project root**:

```bash
python clean_cache.py
```

This removes all `__pycache__` folders and `.pyc` files so Python loads fresh code.

---

## 2. Stop any existing Python / Uvicorn processes

### Windows (PowerShell)

```powershell
# List processes on port 8000
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object OwningProcess

# Kill process by port (replace PID with the number from above, or use one-liner below)
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force -ErrorAction SilentlyContinue

# Or kill all Python processes (use with care if you have other Python apps)
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Windows (CMD)

```cmd
for /f "tokens=5" %a in ('netstat -aon ^| findstr :8000') do taskkill /F /PID %a
```

### One-liner (PowerShell, from project root)

```powershell
Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep -Seconds 2
```

---

## 3. Start Uvicorn on port 8000

From **project root** (`c:\temp2`):

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or without reload:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 4. Verify

- Console should show: `GROQ_API_KEY loaded: True`
- On first `/chat` request: `USING GROQ MODEL: llama-3.1-8b-instant`
- No "decommissioned model" or 404 model errors
