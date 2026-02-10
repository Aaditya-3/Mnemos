"""
Test script to verify environment variable loading
"""

import os
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent
env_path = project_root / ".env"

print(f"Project root: {project_root}")
print(f"Looking for .env at: {env_path}")
print(f".env exists: {env_path.exists()}")

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"Loaded .env from: {env_path}")
else:
    print("ERROR: .env file not found!")
    print("Run: python setup_env.py")

api_key = os.getenv("GROQ_API_KEY")
if api_key:
    print(f"[OK] GROQ_API_KEY loaded: True")
    print(f"[OK] API key preview: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
else:
    print("[ERROR] GROQ_API_KEY loaded: False")
    print("  Make sure .env file exists and contains GROQ_API_KEY=...")
