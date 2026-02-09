"""
Setup helper script to create .env file from .env.example
"""

import shutil
from pathlib import Path

def setup_env():
    """Create .env file from .env.example if it doesn't exist."""
    project_root = Path(__file__).parent
    env_example = project_root / ".env.example"
    env_file = project_root / ".env"
    
    if env_file.exists():
        print(".env file already exists. Skipping setup.")
        return
    
    if not env_example.exists():
        print("ERROR: .env.example not found!")
        return
    
    # Copy .env.example to .env
    shutil.copy(env_example, env_file)
    print("✓ Created .env file from .env.example")
    print("✓ You can now edit .env to add your API key if needed")
    print("✓ The example API key is already in the file")

if __name__ == "__main__":
    setup_env()
