"""
OpenAI 전용: MCP 도구를 Chat Completions tools로 넘기고, tool_calls 시 실행 후 role=tool 메시지로 재전달.
"""
import json
from openai import OpenAI

from agent.tool_registry import openai_tools, execute_tool


SYSTEM_INSTRUCTION = (
    "You are an AI assistant specialized in E-commerce data (orders and customers) and weather data too. "
    "When the user asks a question about orders or customers or weather information, use the provided tools. "
    "If the question is unrelated to your tools, answer using your intrinsic knowledge. "
    "Be concise and do not mention the tools were used unless asked."
)


MAX_TOOL_TURNS = 5


def generate_with_tools(api_key: str, model: str, prompt: str) -> str:
    client = OpenAI(api_key=api_key)
    tools = openai_tools()
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]

    for _ in range(MAX_TOOL_TURNS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not tool_calls:
            return (msg.content or "").strip() or "No response generated."

        messages.append(msg)
        for tc in tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            result = execute_tool(name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
    return "No response generated."
