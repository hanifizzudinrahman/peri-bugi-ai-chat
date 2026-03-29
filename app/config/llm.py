"""
LLM Factory — pluggable provider.

Swap provider cukup dengan ubah LLM_PROVIDER di .env:
- ollama  → Ollama lokal (dev)
- gemini  → Google Gemini (production)
- openai  → OpenAI (production)

Semua provider return BaseChatModel yang interface-nya sama,
sehingga code agent tidak perlu tahu provider yang dipakai.
"""
from langchain_core.language_models import BaseChatModel

from app.config.settings import settings


def get_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
    streaming: bool = True,
) -> BaseChatModel:
    """
    Return LLM instance berdasarkan LLM_PROVIDER di config.
    Parameter opsional override nilai default dari settings.
    """
    temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

    if settings.LLM_PROVIDER == "ollama":
        return _get_ollama(temp, tokens, streaming)
    if settings.LLM_PROVIDER == "gemini":
        return _get_gemini(temp, tokens, streaming)
    if settings.LLM_PROVIDER == "openai":
        return _get_openai(temp, tokens, streaming)

    raise ValueError(f"LLM_PROVIDER tidak dikenal: '{settings.LLM_PROVIDER}'. "
                     f"Pilih: ollama | gemini | openai")


def _get_ollama(temperature: float, max_tokens: int, streaming: bool) -> BaseChatModel:
    from langchain_ollama import ChatOllama
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=temperature,
        num_predict=max_tokens,
        streaming=streaming,
        # Disable thinking mode untuk model seperti qwen3.5, deepseek-r1
        # Tanpa ini, thinking tokens menyebabkan TTFT 60+ detik
        extra_body={"think": False},
    )


def _get_gemini(temperature: float, max_tokens: int, streaming: bool) -> BaseChatModel:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY belum diset di .env")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=temperature,
        max_output_tokens=max_tokens,
        streaming=streaming,
    )


def _get_openai(temperature: float, max_tokens: int, streaming: bool) -> BaseChatModel:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY belum diset di .env")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )


def get_provider_name() -> str:
    """Return nama provider aktif — untuk logging."""
    return settings.LLM_PROVIDER


def get_model_name() -> str:
    """Return nama model aktif — untuk logging."""
    return settings.llm_model_name
