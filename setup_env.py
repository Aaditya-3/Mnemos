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

    shutil.copy(env_example, env_file)
    print("Created .env file from .env.example")
    print("Edit .env to add your Groq and Qdrant credentials before starting the app")


if __name__ == "__main__":
    setup_env()
