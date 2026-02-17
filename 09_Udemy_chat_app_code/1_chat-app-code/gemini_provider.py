from google import genai


class GeminiProvider:
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("API key is required for GeminiProvider")
        if not model_name:
            raise ValueError("Model name is required for GeminiProvider")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate_response(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
            return response.text or "No response generated."
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}") from e
