"""Client wrapper for a local, free LLM via Ollama.

Ollama runs the actual model locally (no API key, no cost, no data leaving
the machine); this class just talks to its local HTTP API. Model choice
resolves the same way as the embedder: constructor arg -> OLLAMA_MODEL env
var -> configs/config.yaml defaults for temperature/max_tokens.
"""
from __future__ import annotations

import logging
import os

import ollama

from clinical_rag.utils.config import load_config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        config = load_config().get("generation", {})

        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = temperature if temperature is not None else config.get("temperature", 0.1)
        self.max_tokens = max_tokens or config.get("max_tokens", 800)

        self.client = ollama.Client(host=self.host)

    def generate(self, prompt: str, temperature: float | None = None) -> str:
        """Sends `prompt` to the local Ollama model and returns the response text.

        Low temperature (0.1 by default) is intentional: this is a factual
        grounding task, not creative writing — we want the model to stick
        closely to the retrieved context rather than improvise. An explicit
        `temperature` override is accepted so a caller retrying after a
        degenerate output can nudge the model onto a different generation
        path rather than resampling the same low-temperature distribution.
        """
        effective_temperature = temperature if temperature is not None else self.temperature
        logger.info("Generating with model=%s (temperature=%s)", self.model, effective_temperature)
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": effective_temperature,
                    "num_predict": self.max_tokens,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reach Ollama at {self.host} with model '{self.model}'. "
                f"Is Ollama running, and has this model been pulled "
                f"(`ollama pull {self.model}`)?"
            ) from exc

        return response["message"]["content"]
