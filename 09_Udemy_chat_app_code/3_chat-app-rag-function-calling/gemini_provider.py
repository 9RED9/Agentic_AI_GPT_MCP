from google import genai
from google.genai import types

from tools import get_current_weather, get_all_customers


def _gemini_tools():
    """Gemini API 스펙: functionDeclarations + parameters (JSON schema)."""
    get_weather = types.FunctionDeclaration(
        name="get_current_weather",
        description="Fetches live weather data for a given location.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or region to fetch weather for, e.g., 'Mumbai India' or 'New York'.",
                },
            },
            "required": ["location"],
        },
    )
    get_customers = types.FunctionDeclaration(
        name="get_all_customers",
        description="Retrieves a list of all registered customers.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of customers to retrieve.",
                },
            },
        },
    )
    return [types.Tool(function_declarations=[get_weather, get_customers])]


def _execute_tool(name: str, args: dict):
    if name == "get_current_weather":
        loc = args.get("location")
        if not isinstance(loc, str):
            raise ValueError("Invalid location")
        return get_current_weather(loc)
    if name == "get_all_customers":
        limit = args.get("limit")
        return get_all_customers(limit)
    raise ValueError(f"Unknown function: {name}")


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

    def generate_embeddings(self, contents, task_type: str = "RETRIEVAL_QUERY"):
        if isinstance(contents, str):
            contents = [contents]
        try:
            response = self._client.models.embed_content(
                model="gemini-embedding-001",
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {e}") from e

    def generate_response_with_tools(self, prompt: str) -> str:
        """Gemini 전용: tools를 넘기고, functionCalls가 있으면 실행 후 결과를 다시 전달."""
        tools = _gemini_tools()
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(tools=tools),
            )
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}") from e

        function_calls = getattr(response, "function_calls", None) or []
        if not function_calls and response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if getattr(part, "function_call", None):
                    function_calls = [part]
                    break
        if function_calls:
            fc = function_calls[0]
            if hasattr(fc, "function_call") and fc.function_call:
                name = getattr(fc.function_call, "name", None)
                raw_args = getattr(fc.function_call, "args", None)
                args = dict(raw_args) if raw_args is not None else {}
            else:
                name = getattr(fc, "name", None)
                raw_args = getattr(fc, "args", None)
                args = dict(raw_args) if raw_args is not None else {}
            if not name:
                return response.text or "No response generated."
            result = _execute_tool(name, args)
            function_response = types.Part.from_function_response(
                name=name,
                response={"result": result},
            )
            follow_up = self._client.models.generate_content(
                model=self._model_name,
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(prompt)]),
                    response.candidates[0].content,
                    types.Content(role="tool", parts=[function_response]),
                ],
                config=types.GenerateContentConfig(tools=tools),
            )
            return follow_up.text or "No response generated."

        return response.text or "No response generated."
