# config/settings.py

import os
from dotenv import load_dotenv

# Ensure environment keys are populated from disk
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama-3.1-8b-instant"
MAX_HANDOVER_CASCADES = 5

# System validation barrier
if not GROQ_API_KEY:
    raise ValueError("CRITICAL CONFIGURATION ERROR: 'GROQ_API_KEY' environment token is missing.")
