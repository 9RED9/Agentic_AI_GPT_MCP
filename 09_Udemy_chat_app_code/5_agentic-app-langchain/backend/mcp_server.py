"""
FastMCP 서버: weather, customer, order, ragSearch 도구 노출.
"""
import os
import json
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from tools.weather import get_weather_data
from tools.customer import get_customers, get_customer_by_id
from tools.order import get_orders, get_orders_with_customer_details, get_order_by_id
from rag.rag_engine import rag_search

load_dotenv()

mcp = FastMCP(
    "agentic-app-langchain",
    stateless_http=True,
    json_response=True,
)


@mcp.tool(
    name="getWeatherData",
    description="Retrieve the live weather data of a city from external API. Get live weather by city or region.",
)
def get_weather_data_tool(city: str, country: str | None = None) -> str:
    return str(get_weather_data(city, country))


@mcp.tool(
    name="getCustomers",
    description="Retrieve all customers from the json data based on limit. Fetch all customers.",
)
def get_customers_tool(limit: int | None = None) -> str:
    return json.dumps(get_customers(limit), indent=2, ensure_ascii=False)


@mcp.tool(
    name="getCustomerById",
    description="Retrieve customer from the json data based on id. Fetch customer by id.",
)
def get_customer_by_id_tool(id: str) -> str:
    c = get_customer_by_id(id)
    return json.dumps(c, indent=2, ensure_ascii=False) if c else "null"


@mcp.tool(
    name="getOrders",
    description="Retrieve all orders from the json data based on limit. Fetch all orders.",
)
def get_orders_tool(limit: int | None = None) -> str:
    return json.dumps(get_orders(limit), indent=2, ensure_ascii=False)


@mcp.tool(
    name="getOrdersWithCustomerDetails",
    description="Retrieve the latest orders along with customer details (name) with optional limit.",
)
def get_orders_with_customer_details_tool(limit: int | None = None) -> str:
    return json.dumps(get_orders_with_customer_details(limit), indent=2, ensure_ascii=False)


@mcp.tool(
    name="getOrderById",
    description="Retrieve order from the json data based on id. Fetch order by id.",
)
def get_order_by_id_tool(id: str) -> str:
    o = get_order_by_id(id)
    return json.dumps(o, indent=2, ensure_ascii=False) if o else "null"


@mcp.tool(
    name="ragSearch",
    description="Retrieve relevant context from vector store for a user query. Use for knowledge-base, company info, policies, product details.",
)
def rag_search_tool(query: str, topK: int = 4) -> str:
    return rag_search(query, top_k=topK)


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8001"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
