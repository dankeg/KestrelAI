import os

import ollama


class LlmWrapper:
    def __init__(
        self, model: str = "gemma3:27b", temperature: float = 0.6, host: str = None
    ):
        self.model = model
        self.temperature = temperature
        # Prefer an explicit host; fall back to env var; then a safe default.
        self.host = host or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)

    def chat(self, messages: list[dict], stream: bool = False) -> str:
        """
        Send chat messages to the LLM and return the response content as a string.

        The Ollama client returns a ChatResponse object with structure:
        - response.message.content (the actual text response)
        """
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=stream,
                options={"temperature": self.temperature},
            )
            if stream:
                return response

            # Ollama returns a ChatResponse object (or dict in some versions)
            # The string content is always at response.message.content
            if hasattr(response, "message"):
                # ChatResponse object: response.message.content
                return response.message.content
            elif isinstance(response, dict) and "message" in response:
                # Dict response: response['message']['content'] or response['message'].content
                message = response["message"]
                if isinstance(message, dict):
                    return message.get("content", "")
                elif hasattr(message, "content"):
                    return message.content

            # Fallback: try direct content access
            if isinstance(response, dict) and "content" in response:
                return response["content"]

            raise ValueError(
                f"Unable to extract content from response. "
                f"Type: {type(response)}, "
                f"Has 'message' attr: {hasattr(response, 'message')}, "
                f"Is dict: {isinstance(response, dict)}, "
                f"Dict keys: {list(response.keys()) if isinstance(response, dict) else 'N/A'}"
            )
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(
                f"LLM chat failed (model: {self.model}, host: {self.host}): {str(e)}"
            ) from e
