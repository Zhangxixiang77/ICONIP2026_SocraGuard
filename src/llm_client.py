"""Unified OpenAI-compatible LLM client.

Every provider in this project (Claude via OpenRouter, GPT-4o via
OpenRouter, DeepSeek, Aliyun DashScope, MiniMax, Zhipu, ...) exposes
an OpenAI-compatible /v1/chat/completions endpoint. We use a single
openai SDK client and just swap base_url + api_key per backend.

Key design choices
------------------
- One LLMClient per LOGICAL backend (deepseek, claude_sonnet, ...).
- All retry / rate-limit logic in one place (tenacity).
- Concurrent dispatch via ThreadPoolExecutor (see `chat_batch`).
- Never silently drop errors — fail loud, fail fast, but with retry.
- Always return the raw assistant text + the model id reported by
  the API (for audit logging).
- A `MockLLMClient` is provided for tests with no API calls.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass
class LLMResponse:
    text: str
    model_reported: str          # what the API claims it is
    backend_name: str            # our logical name (e.g., "deepseek")
    prompt_tokens: int | None
    completion_tokens: int | None
    raw_response: Any = None     # full OpenAI response object, for audit


class LLMClient:
    """One client per logical backend."""

    def __init__(
        self,
        backend_name: str,
        provider: str,
        model_id: str,
        api_key: str,
        base_url: str,
        defaults: dict,
    ):
        self.backend_name = backend_name
        self.provider = provider
        self.model_id = model_id
        self.defaults = defaults
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=defaults.get("timeout_seconds", 90),
        )

    @retry(
        retry=retry_if_exception_type(
            (RateLimitError, APITimeoutError, APIError)
        ),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a chat-completions request and return a normalized response."""
        t = temperature if temperature is not None else self.defaults["temperature"]
        m = max_tokens if max_tokens is not None else self.defaults["max_tokens"]

        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=t,
            max_tokens=m,
        )

        choice = resp.choices[0]
        text = choice.message.content or ""

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            model_reported=getattr(resp, "model", self.model_id) or self.model_id,
            backend_name=self.backend_name,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            raw_response=resp,
        )

    def chat_batch(
        self,
        message_lists: list[list[dict]],
        *,
        max_workers: int = 8,
        on_error: str = "raise",
        **kwargs,
    ) -> list[LLMResponse | None]:
        """Run many chat requests concurrently, preserving order."""
        results: list[LLMResponse | None] = [None] * len(message_lists)

        def _one(i: int):
            try:
                return i, self.chat(message_lists[i], **kwargs), None
            except Exception as e:
                return i, None, e

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_one, i) for i in range(len(message_lists))]
            for fut in as_completed(futures):
                i, resp, err = fut.result()
                if err is not None:
                    if on_error == "raise":
                        raise err
                    if on_error == "placeholder":
                        results[i] = LLMResponse(
                            text="", model_reported=self.model_id,
                            backend_name=self.backend_name,
                            prompt_tokens=None, completion_tokens=None,
                        )
                else:
                    results[i] = resp
        return results


def load_models_config(path: Path | None = None, profile: str | None = None) -> dict:
    """profile: 'mvp' | 'lncs' (default from env SOCRAGUARD_PROFILE or 'mvp')"""
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    if profile is None:
        profile = os.environ.get("SOCRAGUARD_PROFILE", "mvp")
    candidates = [
        CONFIG_DIR / f"models_{profile}.yaml",
        CONFIG_DIR / "models.yaml",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No models config: tried {[str(c) for c in candidates]}")


def load_api_keys(path: Path | None = None) -> dict:
    path = path or (CONFIG_DIR / "api_keys.yaml")
    if not path.exists():
        raise FileNotFoundError(
            f"API keys file not found at {path}. "
            f"Copy api_keys.example.yaml -> api_keys.yaml and fill in keys."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_client(backend_name: str, profile: str | None = None) -> LLMClient:
    models = load_models_config(profile=profile)
    keys = load_api_keys()
    if backend_name not in models["backends"]:
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available: {list(models['backends'])}"
        )
    backend = models["backends"][backend_name]
    provider = backend["provider"]
    if provider not in keys:
        raise ValueError(
            f"No API key for provider '{provider}' (needed by '{backend_name}')."
        )
    return LLMClient(
        backend_name=backend_name,
        provider=provider,
        model_id=backend["model_id"],
        api_key=keys[provider]["api_key"],
        base_url=keys[provider]["base_url"],
        defaults=models["defaults"],
    )


def build_clients(backend_names: list[str], profile: str | None = None) -> dict[str, LLMClient]:
    return {name: build_client(name, profile=profile) for name in backend_names}


class MockLLMClient:
    """Drop-in replacement for LLMClient in tests."""

    def __init__(
        self,
        backend_name: str = "mock",
        provider: str = "mock",
        canned_response: str | Callable[[list[dict]], str] = "OK",
    ):
        self.backend_name = backend_name
        self.provider = provider
        self.model_id = "mock-model"
        self._canned = canned_response
        self.calls: list[dict] = []
        self.defaults = {"temperature": 0.0, "max_tokens": 512, "timeout_seconds": 30}

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        text = self._canned(messages) if callable(self._canned) else self._canned
        return LLMResponse(
            text=text, model_reported="mock-model",
            backend_name=self.backend_name,
            prompt_tokens=10, completion_tokens=5,
        )

    def chat_batch(self, message_lists, **kwargs):
        return [self.chat(ml) for ml in message_lists]
