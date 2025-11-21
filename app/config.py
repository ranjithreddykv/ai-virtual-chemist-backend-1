from dotenv import load_dotenv
from pathlib import Path
import os
from groq import Groq

# Absolute path to .env next to main.py
ENV_PATH = Path(__file__).resolve().parent / ".env"

print("🔍 Looking for .env file at:", ENV_PATH)

if not ENV_PATH.exists():
    raise RuntimeError(f"❌ .env file NOT FOUND at: {ENV_PATH}")

# Load the .env file
load_dotenv(ENV_PATH)

# Read API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY missing in .env!")

# Create Groq client
groq_client = Groq(api_key=GROQ_API_KEY)
