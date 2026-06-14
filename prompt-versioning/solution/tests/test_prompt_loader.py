"""Exercise 2 — environment-aware prompt loader."""
# Tests that load_environment honors PROMPT_ENV and rejects invalid values

import pytest
from src.prompts.loader import load_environment


def test_default_is_prod(monkeypatch):
    monkeypatch.delenv("PROMPT_ENV", raising=False)
    env = load_environment()
    tmpl = env.get_template("docbot_system.j2").render(contexts="")
    assert "[DEV]" not in tmpl


def test_dev_env_loads_dev_template(monkeypatch):
    monkeypatch.setenv("PROMPT_ENV", "dev")
    env = load_environment()
    tmpl = env.get_template("docbot_system.j2").render(contexts="")
    assert "[DEV]" in tmpl


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("PROMPT_ENV", "dev")
    env = load_environment("prod")
    tmpl = env.get_template("docbot_system.j2").render(contexts="")
    assert "[DEV]" not in tmpl


def test_invalid_env_raises():
    with pytest.raises(ValueError):
        load_environment("staging")
