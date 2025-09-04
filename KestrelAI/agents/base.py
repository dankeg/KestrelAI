import os
import ollama


class LlmWrapper:
    def __init__(
        self, model: str = "gemma3:12b", temperature: float = 0.6, host: str = None
    ):
        self.model = model
        self.temperature = temperature
        # Prefer an explicit host; fall back to env var; then a safe default.
        self.client = ollama.Client(
            host=host or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )

    def chat(self, messages: list[dict], stream: bool = False) -> str:
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=stream,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"] if not stream else response
