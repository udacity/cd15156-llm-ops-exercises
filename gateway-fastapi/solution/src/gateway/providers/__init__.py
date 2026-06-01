"""Provider adapters for multi-provider routing — Exercise 3 of Module 18.

The starter routes only OpenAI in production. Adapters live in this package
so the gateway core stays provider-agnostic and new providers can land as
new files under ``src/gateway/providers/`` without touching ``router.py``.
"""

# Package marker — adapters are imported lazily from src/gateway/router.py.
