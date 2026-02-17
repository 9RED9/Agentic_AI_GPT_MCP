"""OpenAI 쪽은 도구 호출 시 Chat Completions API(tools, tool_calls)를 사용한다. Responses API와 스펙이 다름."""
import json
from openai import OpenAI

from tools import get_current_weather, get_all_customers


def _openai_tools():
    """OpenAI API 스펙: tools 배열, type 'function', function.name/description/parameters."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Fetches live weather data for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or region to fetch weather for, e.g., 'Mumbai India' or 'New York'.",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_all_customers",
                "description": "Retrieves a list of all registered customers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of customers to retrieve.",
                        },
                    },
                },
            },
        },
    ]


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


class OpenaiProvider:
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("API key is required for OpenaiProvider")
        if not model_name:
            raise ValueError("Model name is required for OpenaiProvider")
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

    def generate_response(self, prompt: str) -> str:
        try:
            response = self._client.responses.create(
                model=self._model_name,
                input=prompt,
            )
            return response.output_text or "No response generated."
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}") from e

    def generate_embeddings(self, contents):
        if isinstance(contents, str):
            contents = [contents]
        try:
            response = self._client.embeddings.create(
                model="text-embedding-3-small",
                input=contents,
                encoding_format="float",
            )
            return [e.embedding for e in response.data]
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {e}") from e

    def generate_response_with_tools(self, prompt: str) -> str:
        """OpenAI 전용: Chat Completions API에 tools 전달, tool_calls가 있으면 실행 후 결과를 tool 메시지로 재전달."""
        tools = _openai_tools()
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}") from e

        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            return (msg.content or "").strip() or "No response generated."

        tc = tool_calls[0]
        name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")
        result = _execute_tool(name, args)
        messages.append(msg)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

        follow = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        out = follow.choices[0].message.content
        return (out or "").strip() or "No response generated."
