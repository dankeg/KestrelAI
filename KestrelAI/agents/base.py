import os

from KestrelAI.shared.runtime_settings import (
    get_default_model_name,
    resolve_llm_base_url,
)

from .langchain_adapter import LangChainChatAdapter


class LlmWrapper:
    def __init__(
        self,
        model: str = "",
        temperature: float = 0.6,
        host: str = None,
        provider: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or get_default_model_name()
        self.temperature = temperature
        self.provider = provider or os.getenv("LLM_PROVIDER", "openai_compatible")
        self.host = host or resolve_llm_base_url()
        self.api_key = (
            api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        )
        self._adapter = LangChainChatAdapter(
            model=self.model,
            temperature=self.temperature,
            host=self.host,
            provider=self.provider,
            api_key=self.api_key,
        )
        # Keep `.client` for compatibility with existing call sites/tests.
        self.client = self._adapter.client

    def chat(self, messages: list[dict], stream: bool = False) -> str:
        """Send chat messages via the LangChain adapter."""
        try:
            return self._adapter.chat(messages, stream=stream)
        except Exception as e:
            raise RuntimeError(
                f"LLM chat failed (model: {self.model}, provider: {self.provider}, host: {self.host}): {str(e)}"
            ) from e

    def chat_response(self, messages: list[dict]):
        """Send chat messages and retain normalized reasoning/tool metadata."""
        try:
            return self._adapter.chat_response(messages)
        except Exception as e:
            raise RuntimeError(
                f"LLM chat failed (model: {self.model}, provider: {self.provider}, host: {self.host}): {str(e)}"
            ) from e

    def get_capabilities(self) -> dict[str, object]:
        """Return observed runtime capabilities from the adapter."""
        return self._adapter.get_capabilities()
