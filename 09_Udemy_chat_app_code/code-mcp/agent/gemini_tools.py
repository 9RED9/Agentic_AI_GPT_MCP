"""
Gemini 전용: MCP 도구를 FunctionDeclaration으로 넘기고, function_calls 시 실행 후 재전달.
"""
from google import genai
from google.genai import types

from agent.tool_registry import gemini_tool_declarations, execute_tool


SYSTEM_INSTRUCTION = (
    "You are an AI assistant specialized in E-commerce data (orders and customers) and weather data too. "
    "When the user asks a question about orders or customers or weather information, use the provided tools. "
    "If the question is unrelated to your tools, answer using your intrinsic knowledge. "
    "Be concise and do not mention the tools were used unless asked."
)


MAX_TOOL_TURNS = 5


def generate_with_tools(api_key: str, model: str, prompt: str) -> str:
    client = genai.Client(api_key=api_key)
    tools = gemini_tool_declarations()

    contents = [types.Content(role="user", parts=[types.Part.from_text(prompt)])]
    config = types.GenerateContentConfig(
        tools=tools,
        system_instruction=types.Content(
            role="model",
            parts=[types.Part.from_text(SYSTEM_INSTRUCTION)],
        ),
    )

    for _ in range(MAX_TOOL_TURNS):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        if not response.candidates or not response.candidates[0].content.parts:
            return response.text or "No response generated."

        parts = response.candidates[0].content.parts
        function_calls = getattr(response, "function_calls", None) or []
        if not function_calls:
            for part in parts:
                if getattr(part, "function_call", None):
                    function_calls = [part]
                    break

        if not function_calls:
            return response.text or "No response generated."

        fc = function_calls[0]
        if hasattr(fc, "function_call") and fc.function_call:
            name = getattr(fc.function_call, "name", None)
            raw_args = getattr(fc.function_call, "args", None)
            args = dict(raw_args) if raw_args is not None else {}
        else:
            name = getattr(fc, "name", None)
            args = dict(getattr(fc, "args", None) or {})

        if not name:
            return response.text or "No response generated."

        result_str = execute_tool(name, args)
        function_response = types.Part.from_function_response(
            name=name,
            response={"result": result_str},
        )
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(prompt)]),
            response.candidates[0].content,
            types.Content(role="tool", parts=[function_response]),
        ]
    return "No response generated."
