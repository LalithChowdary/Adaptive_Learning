"""Pure-stdlib Cloud LLM HTTP client (OpenAI, Anthropic, Gemini) + deterministic offline doubles."""
import hashlib
import json
import re
import socket
from typing import Any
import urllib.error
import urllib.request

from .config import Settings


class LLMClientError(RuntimeError):
    """Typed error raised on live LLM HTTP, timeout, network, config, or JSON parsing failures."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
        kind: str = "error",
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.kind = kind


def _redact_secret(text: str, secret: str | list[str] | tuple[str, ...]) -> str:
    if not secret:
        return text
    if isinstance(secret, str):
        return text.replace(secret, "[REDACTED]")
    out = text
    for item in secret:
        if item:
            out = out.replace(item, "[REDACTED]")
    return out


def extract_json_object(raw_text: str, *, provider: str = "") -> dict[str, Any]:
    """Parse a JSON object cleanly even when wrapped in ```json ... ``` markdown fences."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise LLMClientError("Empty response text from LLM provider.", provider=provider, kind="invalid_json")

    cleaned = raw_text.strip()
    candidates: list[str] = [cleaned]

    # Extract fenced ```json ... ``` or ``` ... ``` blocks
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE):
        inner = match.group(1).strip()
        if inner:
            candidates.insert(0, inner)

    # Also attempt balanced outermost {...} slice if prose surrounds JSON
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    last_err: Exception | None = None
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        last_err = TypeError(f"Expected JSON object (dict), got {type(parsed).__name__}")

    raise LLMClientError(
        f"Failed to parse valid JSON object from LLM response: {last_err}",
        provider=provider,
        kind="invalid_json",
    )


def _join_endpoint(base_url: str, suffix: str, *, version_prefix: str = "") -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise LLMClientError("LLM base URL is empty.", kind="config_error")
    if base.endswith(suffix):
        return base
    if version_prefix and not base.endswith(version_prefix):
        # Allow callers to supply either root (http://127.0.0.1:PORT) or versioned base (.../v1)
        return f"{base}{version_prefix}{suffix}"
    return f"{base}{suffix}"


def _extract_prompt_context(user_prompt: str) -> dict[str, Any]:
    """Best-effort parse of structured context JSON embedded in user_prompt for offline doubles."""
    if not user_prompt:
        return {}
    try:
        parsed = json.loads(user_prompt)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    first = user_prompt.find("{")
    last = user_prompt.rfind("}")
    if first != -1 and last > first:
        try:
            parsed = json.loads(user_prompt[first : last + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


class LLMClient:
    """Stdlib HTTP client for OpenAI, Anthropic, and Gemini with offline deterministic doubles."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self._api_key = self.settings.llm_api_key
        secrets = [self._api_key]
        for _, key, _, _ in getattr(self.settings, "provider_configs", ()):
            if key and key not in secrets:
                secrets.append(key)
        self._secrets: tuple[str, ...] = tuple(s for s in secrets if s)
        self._last_telemetry: dict[str, Any] = {
            "provider": self.provider_name(),
            "model": self.model_name(),
            "failover_used": False,
            "attempts": [],
        }

    def __repr__(self) -> str:
        return (
            f"LLMClient(provider={self.provider_name()!r}, "
            f"model={self.model_name()!r}, is_live={self.is_live()!r})"
        )

    def is_live(self) -> bool:
        return self.settings.is_live()

    def provider_name(self) -> str:
        return self.settings.effective_provider

    def model_name(self) -> str:
        return self.settings.llm_model

    def last_call_info(self) -> dict[str, Any]:
        return {
            "provider": self._last_telemetry.get("provider", self.provider_name()),
            "model": self._last_telemetry.get("model", self.model_name()),
            "failover_used": bool(self._last_telemetry.get("failover_used", False)),
            "attempts": list(self._last_telemetry.get("attempts", [])),
        }

    def _provider_chain(self) -> list[tuple[str, str, str, str]]:
        configs = list(getattr(self.settings, "provider_configs", ()) or ())
        if configs:
            return configs
        return [
            (
                self.settings.effective_provider,
                self._api_key,
                self.settings.llm_model,
                self.settings.llm_base_url,
            )
        ]

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str = "generic",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1800,
    ) -> dict[str, Any]:
        """Execute a structured JSON completion with ordered provider failover or offline double."""
        if not self.is_live():
            res = self._deterministic_double(system_prompt, user_prompt, schema_name)
            self._last_telemetry = {
                "provider": "mock",
                "model": self.model_name(),
                "failover_used": False,
                "attempts": [{"provider": "mock", "status": "ok"}],
            }
            return res

        attempts: list[dict[str, Any]] = []
        last_exc: LLMClientError | None = None
        chain = self._provider_chain()
        for idx, (provider, api_key, model, base_url) in enumerate(chain):
            if provider == "mock":
                res = self._deterministic_double(system_prompt, user_prompt, schema_name)
                attempts.append({"provider": "deterministic_double", "status": "ok"})
                self._last_telemetry = {
                    "provider": "deterministic_double",
                    "model": model or "deterministic-double-v1",
                    "failover_used": idx > 0,
                    "attempts": attempts,
                }
                return res
            try:
                if provider in ("openai", "groq", "ollama"):
                    raw_text = self._call_openai(
                        system_prompt,
                        user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                        provider=provider,
                    )
                elif provider == "anthropic":
                    raw_text = self._call_anthropic(
                        system_prompt,
                        user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                elif provider == "gemini":
                    raw_text = self._call_gemini(
                        system_prompt,
                        user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                else:
                    raise LLMClientError(f"Unsupported live provider {provider!r}.", provider=provider, kind="config_error")

                parsed = extract_json_object(raw_text, provider=provider)
                attempts.append({"provider": provider, "model": model, "status": "ok"})
                self._last_telemetry = {
                    "provider": provider,
                    "model": model,
                    "failover_used": idx > 0,
                    "attempts": attempts,
                }
                return parsed
            except LLMClientError as exc:
                last_exc = exc
                attempts.append(
                    {
                        "provider": provider,
                        "model": model,
                        "status": "error",
                        "kind": exc.kind,
                        "status_code": exc.status_code,
                    }
                )
                continue

        self._last_telemetry = {
            "provider": self.provider_name(),
            "model": self.model_name(),
            "failover_used": len(attempts) > 1,
            "attempts": attempts,
        }
        assert last_exc is not None
        raise last_exc

    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        provider: str,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req_headers = {"User-Agent": "adaptive-web/1.0", **headers}
        req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
        timeout = float(self.settings.llm_timeout_seconds)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")[:400]
            except Exception:
                err_body = ""
            msg = _redact_secret(
                f"HTTP {exc.code} from {provider}: {err_body or exc.reason}",
                self._secrets,
            )
            raise LLMClientError(msg, provider=provider, status_code=int(exc.code), kind="http_error") from exc
        except (TimeoutError, socket.timeout) as exc:
            msg = _redact_secret(f"Request to {provider} timed out after {timeout:.1f}s.", self._secrets)
            raise LLMClientError(msg, provider=provider, kind="timeout") from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, (TimeoutError, socket.timeout)):
                msg = _redact_secret(f"Request to {provider} timed out after {timeout:.1f}s.", self._secrets)
                raise LLMClientError(msg, provider=provider, kind="timeout") from exc
            msg = _redact_secret(f"Network error contacting {provider}: {reason}", self._secrets)
            raise LLMClientError(msg, provider=provider, kind="network_error") from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                f"Provider {provider} returned non-JSON HTTP body: {exc}",
                provider=provider,
                kind="invalid_json",
            ) from exc
        if not isinstance(decoded, dict):
            raise LLMClientError(
                f"Provider {provider} returned unexpected JSON type {type(decoded).__name__}.",
                provider=provider,
                kind="invalid_json",
            )
        return decoded

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        provider: str = "openai",
    ) -> str:
        url = _join_endpoint(base_url or self.settings.llm_base_url, "/chat/completions", version_prefix="/v1")
        used_model = model or self.model_name()
        used_key = api_key if api_key is not None else self._api_key
        payload = {
            "model": used_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {used_key}",
        }
        data = self._post_json(url, payload, headers, provider=provider)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(
                f"Unexpected {provider} response structure: {exc}",
                provider=provider,
                kind="invalid_json",
            ) from exc
        if isinstance(content, list):
            parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            return "\n".join(parts)
        return str(content)

    def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> str:
        url = _join_endpoint(base_url or self.settings.llm_base_url, "/messages", version_prefix="/v1")
        used_model = model or self.model_name()
        used_key = api_key if api_key is not None else self._api_key
        payload = {
            "model": used_model,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": used_key,
            "anthropic-version": "2023-06-01",
        }
        data = self._post_json(url, payload, headers, provider="anthropic")
        content = data.get("content")
        if not isinstance(content, list) or not content:
            raise LLMClientError(
                "Unexpected Anthropic response structure: missing 'content' blocks.",
                provider="anthropic",
                kind="invalid_json",
            )
        texts = [
            str(block["text"])
            for block in content
            if isinstance(block, dict) and block.get("type", "text") == "text" and "text" in block
        ]
        if not texts:
            raise LLMClientError(
                "Anthropic response contained no text blocks.",
                provider="anthropic",
                kind="invalid_json",
            )
        return "\n".join(texts)

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> str:
        base = (base_url or self.settings.llm_base_url).strip().rstrip("/")
        used_model = model or self.model_name()
        used_key = api_key if api_key is not None else self._api_key
        if base.endswith(":generateContent"):
            url = base
        elif base.endswith("/v1beta"):
            url = f"{base}/models/{used_model}:generateContent"
        else:
            url = f"{base}/v1beta/models/{used_model}:generateContent"

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
                "responseMimeType": "application/json",
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": used_key,
        }
        data = self._post_json(url, payload, headers, provider="gemini")
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(
                f"Unexpected Gemini response structure: {exc}",
                provider="gemini",
                kind="invalid_json",
            ) from exc
        if not isinstance(parts, list) or not parts:
            raise LLMClientError(
                "Gemini response contained empty 'parts'.",
                provider="gemini",
                kind="invalid_json",
            )
        texts = [str(p["text"]) for p in parts if isinstance(p, dict) and "text" in p]
        if not texts:
            raise LLMClientError(
                "Gemini response contained no text parts.",
                provider="gemini",
                kind="invalid_json",
            )
        return "\n".join(texts)

    def _deterministic_double(self, system_prompt: str, user_prompt: str, schema_name: str) -> dict[str, Any]:
        ctx = _extract_prompt_context(user_prompt)
        digest = hashlib.sha256(f"{schema_name}:{system_prompt}:{user_prompt}".encode("utf-8")).hexdigest()[:12]
        return {
            "schema_name": schema_name,
            "double_id": digest,
            "context_echo": ctx,
            "source": "deterministic_double",
        }
