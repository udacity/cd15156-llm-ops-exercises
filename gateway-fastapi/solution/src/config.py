"""Application settings for the ScikitDocs starter.

Loaded from environment variables and `.env`. All default values come
from `src/constants.py` — change them there, not here.
"""

from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from src import constants

# Load .env into os.environ early so libraries that read environment
# variables directly (the openai SDK's base_url discovery, RAGAS's
# internal OpenAI client) see the project's keys without needing a
# manual `set -a; source .env` step. Matches capstone behavior.
load_dotenv()


class Settings(BaseSettings):
    """Central configuration — all values come from .env or the environment."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # === API keys ===
    openai_api_key: str = ""
    # Empty string → use the SDK default (https://api.openai.com/v1).
    # Set to constants.VOCAREUM_BASE_URL when running with a `voc-` key.
    openai_base_url: str = ""

    # === Chroma ===
    chroma_path: str = "data/chroma"

    # === Tracing — REQ-067 (M09) populates this ===
    tracing_backend: Literal["phoenix", "none"] = "phoenix"
    phoenix_embedded: bool = True
    phoenix_host: str = "0.0.0.0"
    phoenix_port: int = constants.PHOENIX_PORT
    phoenix_working_dir: str = "data/phoenix"
    phoenix_project_name: str = constants.PHOENIX_PROJECT_NAME

    # === Models ===
    model_complex: str = constants.MODEL_COMPLEX
    model_simple: str = constants.MODEL_SIMPLE
    # Exercise 1 — third tier. Placeholder points at gpt-4o so the dispatch
    # stays Vocareum-compatible; swap to gpt-4-turbo or claude-sonnet outside
    # the sandbox.
    model_premium: str = "gpt-4o"
    embedding_model: str = constants.EMBEDDING_MODEL

    # === Application ===
    confidence_threshold: float = constants.CONFIDENCE_THRESHOLD
    cost_log_path: str = constants.COST_LOG_PATH


settings = Settings()
