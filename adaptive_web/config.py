"""Environment and .env configuration for 3/adaptive_web/ (pure stdlib, no secrets leaked)."""
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from typing import Any

from . import PACKAGE_DIR, REPO_ROOT

VALID_PROVIDERS = ("auto", "openai", "groq", "anthropic", "gemini", "ollama", "mock")
VALID_FALLBACK_ITEMS = ("openai", "groq", "anthropic", "gemini", "ollama", "mock", "deterministic_double")
VALID_DEVICES = ("cpu", "mps", "auto")

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq": "openai/gpt-oss-120b",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.1",
    "mock": "deterministic-double-v1",
}

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://127.0.0.1:11434/v1",
    "mock": "",
}

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _unescape_double_quoted(raw: str) -> str:
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            if nxt == "n":
                out.append("\n")
            elif nxt == "r":
                out.append("\r")
            elif nxt == "t":
                out.append("\t")
            elif nxt in ('"', "\\"):
                out.append(nxt)
            else:
                out.append(nxt)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _parse_env_value(raw: str) -> str:
    val = raw.strip()
    if not val:
        return ""
    quote = val[0]
    if quote in ('"', "'"):
        i = 1
        n = len(val)
        while i < n:
            if quote == '"' and val[i] == "\\" and i + 1 < n:
                i += 2
                continue
            if val[i] == quote:
                inner = val[1:i]
                return _unescape_double_quoted(inner) if quote == '"' else inner
            i += 1
        # Unclosed quote fallback: return stripped content after opening quote
        return val[1:].strip()
    # Unquoted value: strip inline # comments preceded by whitespace
    return re.split(r"\s+#", val, maxsplit=1)[0].strip()


def parse_dotenv(path: str | Path) -> dict[str, str]:
    """Parse a .env file using pure Python stdlib without mutating os.environ."""
    p = Path(path)
    if not p.is_file():
        return {}
    parsed: dict[str, str] = {}
    text = p.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export ") and len(stripped) > 7:
            stripped = stripped[7:].lstrip()
        elif stripped.startswith("export\t") and len(stripped) > 7:
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue
        key, raw_val = stripped.split("=", 1)
        key = key.strip()
        if not _KEY_RE.match(key):
            continue
        parsed[key] = _parse_env_value(raw_val)
    return parsed


def _infer_provider(model: str, base_url: str) -> str:
    m = model.lower()
    b = base_url.lower()
    if "claude" in m or "anthropic" in b:
        return "anthropic"
    if "gemini" in m or "generativelanguage" in b or "gemini" in b:
        return "gemini"
    if "groq" in b:
        return "groq"
    if "11434" in b or "ollama" in b:
        return "ollama"
    return "openai"


@dataclass(frozen=True)
class Settings:
    adaptive_web_port: int = 8780
    adaptive_web_db: Path = PACKAGE_DIR / "data" / "adaptive_web.sqlite3"
    llm_provider: str = "auto"
    effective_provider: str = "mock"
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = "deterministic-double-v1"
    llm_base_url: str = ""
    llm_timeout_seconds: float = 45.0
    device: str = "cpu"
    env_source: str = "defaults"
    fallback_providers: tuple[str, ...] = field(default_factory=tuple)
    provider_configs: tuple[tuple[str, str, str, str], ...] = field(default_factory=tuple, repr=False)

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "Settings":
        merged: dict[str, str] = {}
        sources: list[str] = []

        if env_path is not None:
            p = Path(env_path)
            loaded = parse_dotenv(p)
            if loaded:
                merged.update(loaded)
                sources.append(str(p))
        else:
            for candidate in (REPO_ROOT / ".env", PACKAGE_DIR / ".env"):
                loaded = parse_dotenv(candidate)
                if loaded:
                    merged.update(loaded)
                    sources.append(str(candidate))

        # os.environ always overrides .env files without mutating os.environ
        relevant_keys = (
            "ADAPTIVE_WEB_PORT",
            "ADAPTIVE_WEB_DB",
            "LLM_PROVIDER",
            "LLM_FALLBACK_PROVIDERS",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_BASE_URL",
            "GROQ_API_KEY",
            "GROQ_MODEL",
            "GROQ_BASE_URL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_MODEL",
            "ANTHROPIC_BASE_URL",
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "GEMINI_BASE_URL",
            "OLLAMA_API_KEY",
            "OLLAMA_MODEL",
            "OLLAMA_BASE_URL",
            "LLM_MODEL",
            "LLM_BASE_URL",
            "LLM_TIMEOUT_SECONDS",
            "DEVICE",
        )
        env_used = False
        for k in relevant_keys:
            if k in os.environ:
                merged[k] = os.environ[k]
                env_used = True
        if env_used:
            sources.append("os.environ")

        def get(key: str, default: str = "") -> str:
            return merged.get(key, default)

        port_raw = get("ADAPTIVE_WEB_PORT", "8780").strip() or "8780"
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid ADAPTIVE_WEB_PORT: {port_raw!r}") from exc
        if not (0 <= port <= 65535):
            raise ValueError(f"ADAPTIVE_WEB_PORT out of range [0, 65535]: {port}")

        db_raw = get("ADAPTIVE_WEB_DB", "").strip()
        if not db_raw:
            db_path = PACKAGE_DIR / "data" / "adaptive_web.sqlite3"
        elif db_raw == ":memory:":
            db_path = Path(":memory:")
        else:
            candidate_db = Path(db_raw).expanduser()
            db_path = candidate_db if candidate_db.is_absolute() else (REPO_ROOT / candidate_db).resolve()

        provider = (get("LLM_PROVIDER", "auto").strip().lower() or "auto")
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Invalid LLM_PROVIDER {provider!r}; expected one of {VALID_PROVIDERS}")

        timeout_raw = get("LLM_TIMEOUT_SECONDS", "45.0").strip() or "45.0"
        try:
            timeout = float(timeout_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid LLM_TIMEOUT_SECONDS: {timeout_raw!r}") from exc
        if not (math.isfinite(timeout) and timeout > 0.0):
            raise ValueError(f"LLM_TIMEOUT_SECONDS must be finite and > 0, got {timeout}")

        device = (get("DEVICE", "cpu").strip().lower() or "cpu")
        if device not in VALID_DEVICES:
            raise ValueError(f"Invalid DEVICE {device!r}; expected one of {VALID_DEVICES}")

        raw_model = get("LLM_MODEL", "").strip()
        raw_base_url = get("LLM_BASE_URL", "").strip()

        generic_key = get("LLM_API_KEY", "").strip()
        provider_keys = {
            "openai": get("OPENAI_API_KEY", "").strip(),
            "groq": get("GROQ_API_KEY", "").strip(),
            "anthropic": get("ANTHROPIC_API_KEY", "").strip(),
            "gemini": get("GEMINI_API_KEY", "").strip(),
            "ollama": get("OLLAMA_API_KEY", "").strip(),
        }
        provider_models = {
            "openai": get("OPENAI_MODEL", "").strip(),
            "groq": get("GROQ_MODEL", "").strip(),
            "anthropic": get("ANTHROPIC_MODEL", "").strip(),
            "gemini": get("GEMINI_MODEL", "").strip(),
            "ollama": get("OLLAMA_MODEL", "").strip(),
        }
        provider_urls = {
            "openai": get("OPENAI_BASE_URL", "").strip(),
            "groq": get("GROQ_BASE_URL", "").strip(),
            "anthropic": get("ANTHROPIC_BASE_URL", "").strip(),
            "gemini": get("GEMINI_BASE_URL", "").strip(),
            "ollama": get("OLLAMA_BASE_URL", "").strip(),
        }

        if provider == "mock":
            effective_provider = "mock"
            api_key = ""
        elif provider == "ollama":
            effective_provider = "ollama"
            api_key = generic_key or provider_keys["ollama"] or "ollama"
        elif provider in ("openai", "groq", "anthropic", "gemini"):
            effective_provider = provider
            api_key = generic_key or provider_keys[provider]
        else:  # provider == "auto"
            if generic_key:
                effective_provider = _infer_provider(raw_model, raw_base_url)
                api_key = generic_key
            else:
                effective_provider = "mock"
                api_key = ""
                for candidate_prov in ("openai", "groq", "anthropic", "gemini"):
                    if provider_keys[candidate_prov]:
                        effective_provider = candidate_prov
                        api_key = provider_keys[candidate_prov]
                        break
                if effective_provider == "mock" and (provider_urls["ollama"] or provider_models["ollama"]):
                    effective_provider = "ollama"
                    api_key = provider_keys["ollama"] or "ollama"

        llm_model = raw_model or provider_models.get(effective_provider) or DEFAULT_MODELS[effective_provider]
        llm_base_url = raw_base_url or provider_urls.get(effective_provider) or DEFAULT_BASE_URLS[effective_provider]

        # Build ordered multi-provider fallback chain from LLM_FALLBACK_PROVIDERS
        raw_fallback = get("LLM_FALLBACK_PROVIDERS", "").strip()
        fallback_items: list[str] = []
        for item in (p.strip().lower() for p in raw_fallback.split(",") if p.strip()):
            if item not in VALID_FALLBACK_ITEMS:
                raise ValueError(f"Invalid fallback provider {item!r}; expected one of {VALID_FALLBACK_ITEMS}")
            if item not in fallback_items:
                fallback_items.append(item)

        chain_configs: list[tuple[str, str, str, str]] = []
        seen_provs: set[str] = set()
        if effective_provider != "mock" and api_key:
            chain_configs.append((effective_provider, api_key, llm_model, llm_base_url))
            seen_provs.add(effective_provider)

        public_fallbacks: list[str] = []
        for fb in fallback_items:
            canonical_fb = "mock" if fb == "deterministic_double" else fb
            if canonical_fb in seen_provs:
                continue
            if canonical_fb == "mock":
                chain_configs.append(("mock", "", DEFAULT_MODELS["mock"], ""))
                seen_provs.add("mock")
                public_fallbacks.append("deterministic_double")
            elif canonical_fb == "ollama":
                fb_key = provider_keys["ollama"] or "ollama"
                fb_model = provider_models["ollama"] or DEFAULT_MODELS["ollama"]
                fb_url = provider_urls["ollama"] or DEFAULT_BASE_URLS["ollama"]
                chain_configs.append(("ollama", fb_key, fb_model, fb_url))
                seen_provs.add("ollama")
                public_fallbacks.append("ollama")
            else:
                fb_key = provider_keys.get(canonical_fb) or (generic_key if canonical_fb == effective_provider else "")
                if fb_key:
                    fb_model = provider_models.get(canonical_fb) or DEFAULT_MODELS[canonical_fb]
                    fb_url = provider_urls.get(canonical_fb) or DEFAULT_BASE_URLS[canonical_fb]
                    chain_configs.append((canonical_fb, fb_key, fb_model, fb_url))
                    seen_provs.add(canonical_fb)
                    public_fallbacks.append(canonical_fb)

        env_source = " + ".join(sources) if sources else "defaults"

        return cls(
            adaptive_web_port=port,
            adaptive_web_db=db_path,
            llm_provider=provider,
            effective_provider=effective_provider,
            llm_api_key=api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_timeout_seconds=timeout,
            device=device,
            env_source=env_source,
            fallback_providers=tuple(public_fallbacks),
            provider_configs=tuple(chain_configs),
        )

    def is_live(self) -> bool:
        if self.effective_provider != "mock" and bool(self.llm_api_key):
            return True
        return any(prov != "mock" and bool(key) for prov, key, _, _ in self.provider_configs)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "adaptive_web_port": self.adaptive_web_port,
            "adaptive_web_db": str(self.adaptive_web_db),
            "llm_provider": self.llm_provider,
            "effective_provider": self.effective_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "device": self.device,
            "env_source": self.env_source,
            "fallback_providers": list(self.fallback_providers),
            "is_live": self.is_live(),
            "api_key_configured": bool(self.llm_api_key) or any(bool(k) for p, k, _, _ in self.provider_configs if p != "mock"),
        }
