"""
MCP server with FastMCP (streamable-http). Exposes weather, customer, order tools.
Run: uvicorn mcp_server:app --host 0.0.0.0 --port 8001
Or: python -m mcp_server (if __main__ runs FastMCP)
"""
import os
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from tools.weather import get_weather_data
from tools.customer import get_customers, get_customer_by_id
from tools.order import get_orders, get_orders_with_customer_details, get_order_by_id

load_dotenv()

mcp = FastMCP(
    "agentic-app",
    stateless_http=True,
    json_response=True,
)


@mcp.tool(
    name="getWeatherData",
    description="Retrieve the live weather data of a city from external API. Get live weather by city or region.",
)
def get_weather_data_tool(city: str, country: str | None = None) -> str:
    result = get_weather_data(city, country)
    return str(result)


@mcp.tool(
    name="getCustomers",
    description="Retrieve all customers from the json data based on limit. Fetch all customers.",
)
def get_customers_tool(limit: int | None = None) -> str:
    import json
    return json.dumps(get_customers(limit), indent=2, ensure_ascii=False)


@mcp.tool(
    name="getCustomerById",
    description="Retrieve customer from the json data based on id. Fetch customer by id.",
)
def get_customer_by_id_tool(id: str) -> str:
    import json
    c = get_customer_by_id(id)
    return json.dumps(c, indent=2, ensure_ascii=False) if c else "null"


@mcp.tool(
    name="getOrders",
    description="Retrieve all orders from the json data based on limit. Fetch all orders.",
)
def get_orders_tool(limit: int | None = None) -> str:
    import json
    return json.dumps(get_orders(limit), indent=2, ensure_ascii=False)


@mcp.tool(
    name="getOrdersWithCustomerDetails",
    description="Retrieve the latest orders along with customer details (name) with optional limit. Fetch all orders with customer details.",
)
def get_orders_with_customer_details_tool(limit: int | None = None) -> str:
    import json
    return json.dumps(get_orders_with_customer_details(limit), indent=2, ensure_ascii=False)


@mcp.tool(
    name="getOrderById",
    description="Retrieve order from the json data based on id. Fetch order by id.",
)
def get_order_by_id_tool(id: str) -> str:
    import json
    o = get_order_by_id(id)
    return json.dumps(o, indent=2, ensure_ascii=False) if o else "null"


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8001"))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
