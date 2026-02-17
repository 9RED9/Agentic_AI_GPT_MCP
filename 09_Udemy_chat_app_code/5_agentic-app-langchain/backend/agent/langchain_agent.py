"""
LangChain 기반 에이전트: ChatGoogleGenerativeAI / ChatOpenAI에 도구를 bind_tools로 붙이고,
tool_calls 반복 실행 후 최종 응답 반환.
"""
import json
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from tools.weather import get_weather_data
from tools.customer import get_customers, get_customer_by_id
from tools.order import get_orders, get_orders_with_customer_details, get_order_by_id
from rag.rag_engine import rag_search


@tool
def getWeatherData(city: str, country: str | None = None) -> str:
    """Retrieve the live weather data of a city from external API. Get live weather by city or region."""
    result = get_weather_data(city, country)
    return json.dumps(result, indent=2, ensure_ascii=False)


@tool
def getCustomers(limit: int | None = None) -> str:
    """Retrieve all customers from the json data based on limit. Fetch all customers."""
    return json.dumps(get_customers(limit), indent=2, ensure_ascii=False)


@tool
def getCustomerById(id: str) -> str:
    """Retrieve customer from the json data based on id. Fetch customer by id."""
    c = get_customer_by_id(id)
    return json.dumps(c, indent=2, ensure_ascii=False) if c else "null"


@tool
def getOrders(limit: int | None = None) -> str:
    """Retrieve all orders from the json data based on limit. Fetch all orders."""
    return json.dumps(get_orders(limit), indent=2, ensure_ascii=False)


@tool
def getOrdersWithCustomerDetails(limit: int | None = None) -> str:
    """Retrieve the latest orders along with customer details (name) with optional limit."""
    return json.dumps(get_orders_with_customer_details(limit), indent=2, ensure_ascii=False)


@tool
def getOrderById(id: str) -> str:
    """Retrieve order from the json data based on id. Fetch order by id."""
    o = get_order_by_id(id)
    return json.dumps(o, indent=2, ensure_ascii=False) if o else "null"


@tool
def ragSearch(query: str, topK: int = 4) -> str:
    """Retrieve relevant context from vector store for a user query. Use for knowledge-base, company info, policies, product details."""
    return rag_search(query, top_k=topK)


LANGCHAIN_TOOLS = [
    getWeatherData,
    getCustomers,
    getCustomerById,
    getOrders,
    getOrdersWithCustomerDetails,
    getOrderById,
    ragSearch,
]

SYSTEM_INSTRUCTION = """You are an AI assistant with access to internal tools.

TOOLS:
- RAG Search tool for knowledge-base, company information, policies, product details, and stored documents.
- Other internal tools provided to you.

BEHAVIOR RULES:
1. When a question is clearly related to stored documents, company information, product details, order/refund/shipping/support topics, or any knowledge-base content, try using the RAG Search tool first.
2. If the RAG tool returns no useful information, or the query is unrelated to stored documents, answer using your own general knowledge.
3. Use other tools only when they are clearly relevant to the user's question.
4. If no tool is relevant, answer directly using your own reasoning.
5. Do not mention tool usage unless the user specifically asks.
6. Keep responses concise, accurate, and helpful.
"""

MAX_TOOL_TURNS = 5


def _tool_by_name(name: str):
    m = {t.name: t for t in LANGCHAIN_TOOLS}
    return m.get(name)


def generate_gemini(api_key: str, model: str, prompt: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(LANGCHAIN_TOOLS)
    messages = [
        HumanMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=prompt),
    ]
    for _ in range(MAX_TOOL_TURNS):
        response = llm_with_tools.invoke(messages)
        if not getattr(response, "tool_calls", None):
            return (response.content or "").strip() or "No response generated."
        messages.append(response)
        for tc in response.tool_calls:
            tool_fn = _tool_by_name(tc["name"])
            if not tool_fn:
                out = f"Unknown tool: {tc['name']}"
            else:
                args = tc.get("args") or {}
                out = tool_fn.invoke(args)
            messages.append(
                ToolMessage(content=str(out), tool_call_id=tc.get("id", ""))
            )
    return "No response generated."


def generate_openai(api_key: str, model: str, prompt: str) -> str:
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)
    llm_with_tools = llm.bind_tools(LANGCHAIN_TOOLS)
    messages = [
        HumanMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=prompt),
    ]
    for _ in range(MAX_TOOL_TURNS):
        response = llm_with_tools.invoke(messages)
        if not getattr(response, "tool_calls", None):
            return (response.content or "").strip() or "No response generated."
        messages.append(response)
        for tc in response.tool_calls:
            tool_fn = _tool_by_name(tc["name"])
            if not tool_fn:
                out = f"Unknown tool: {tc['name']}"
            else:
                args = tc.get("args") or {}
                out = tool_fn.invoke(args)
            messages.append(
                ToolMessage(content=str(out), tool_call_id=tc.get("id", ""))
            )
    return "No response generated."
