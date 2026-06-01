"""Environment-aware prompt loader.

Reads PROMPT_ENV from the environment (defaults to "prod") and
returns a Jinja2 Environment rooted at prompts/<env>/.
"""
# TODO(m03-ex2): build env-aware prompt loader (whole file)
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]

def load_environment(env_name: str | None = None) -> Environment:
    env_name = env_name or os.environ.get("PROMPT_ENV", "prod")
    if env_name not in ("dev", "prod"):
        raise ValueError(f"PROMPT_ENV must be 'dev' or 'prod', got {env_name!r}")
    prompts_dir = _REPO_ROOT / "prompts" / env_name
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"prompts directory not found: {prompts_dir}")
    return Environment(
        loader=FileSystemLoader(prompts_dir),
        keep_trailing_newline=True,
        autoescape=False,
    )
