"""
Gemini LLM Client

Wrapper for Google Gemini API using google-generativeai SDK.
"""

import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from project root
# Try multiple paths to ensure we find .env file
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

# Try loading from multiple locations (same logic as backend)
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(dotenv_path=cwd_env, override=True)
    else:
        load_dotenv(override=True)


def call_gemini(prompt: str, memory_context: str = "") -> str:
    """
    Call Gemini API with the given prompt.
    
    Args:
        prompt: User message or prompt
        memory_context: Optional memory context to include in the prompt
    
    Returns:
        Generated response text
    
    Raises:
        RuntimeError: If API call fails or API key is missing
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment variables")
    
    try:
        # Initialize Gemini inside function (not at module level)
        genai.configure(api_key=api_key)
        
        # Use gemini-1.5-flash model (NO system_instruction parameter)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Merge memory context and user prompt into a single string
        # Make memory context VERY prominent so Gemini uses it
        full_prompt = prompt
        if memory_context:
            full_prompt = f"""IMPORTANT CONTEXT ABOUT THE USER (remember and use this information):
{memory_context}

Now respond to the user's message below. Use the context above to personalize your response and remember who you're talking to.

User: {prompt}"""
        
        # Generate response with the combined prompt
        response = model.generate_content(full_prompt)
        
        if not response or not response.text:
            raise RuntimeError("Empty response from Gemini")
        
        return response.text
        
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {str(e)}")
