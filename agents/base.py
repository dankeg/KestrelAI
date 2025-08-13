import ollama

class LlmWrapper:
    def __init__(self, model: str = "gemma3:12b", temperature: float = 0.6):
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list[dict], stream: bool = False) -> str:
        response = ollama.chat(
            model=self.model,
            messages=messages,
            stream=stream,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"] if not stream else response
