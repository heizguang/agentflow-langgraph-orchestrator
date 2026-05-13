import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def _normalize_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if normalized and not normalized.endswith("/v1"):
        normalized = normalized + "/v1"
    return normalized


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def get_default_model() -> str:
    return (
        os.getenv("AGENTFLOW_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or "gpt-5.4"
    )


def build_chat_model(
    temperature: float = 0.0,
    model: str | None = None,
    timeout: int = 180,
):
    api_key = _get_required_env("OPENAI_API_KEY")
    base_url = _normalize_base_url(_get_required_env("OPENAI_BASE_URL"))
    chosen_model = (model or "").strip() or get_default_model()
    return ChatOpenAI(
        model=chosen_model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=2,
    )
