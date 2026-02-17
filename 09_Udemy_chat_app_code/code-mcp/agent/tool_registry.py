"""
MCP와 동일한 도구 스키마 및 실행 함수. 에이전트는 이 레지스트리로 Gemini/OpenAI에 도구를 넘기고, 호출 시 실행한다.
"""
import json
from tools.weather import get_weather_data
from tools.customer import get_customers, get_customer_by_id
from tools.order import get_orders, get_orders_with_customer_details, get_order_by_id


def _run(name: str, args: dict):
    if name == "getWeatherData":
        return get_weather_data(args.get("city", ""), args.get("country"))
    if name == "getCustomers":
        return get_customers(args.get("limit"))
    if name == "getCustomerById":
        return get_customer_by_id(args.get("id", ""))
    if name == "getOrders":
        return get_orders(args.get("limit"))
    if name == "getOrdersWithCustomerDetails":
        return get_orders_with_customer_details(args.get("limit"))
    if name == "getOrderById":
        return get_order_by_id(args.get("id", ""))
    raise ValueError(f"Unknown tool: {name}")


def execute_tool(name: str, arguments: dict) -> str:
    result = _run(name, arguments)
    return json.dumps(result, indent=2, ensure_ascii=False)


# Gemini용 FunctionDeclaration 스키마 (google.genai.types)
def gemini_tool_declarations():
    from google.genai import types
    return [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="getWeatherData",
                description="Retrieve the live weather data of a city from external API. Get live weather by city or region.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "country": {"type": "string", "description": "Country name (optional)"},
                    },
                    "required": ["city"],
                },
            ),
            types.FunctionDeclaration(
                name="getCustomers",
                description="Retrieve all customers from the json data based on limit. Fetch all customers.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "description": "Maximum number of customers"}},
                },
            ),
            types.FunctionDeclaration(
                name="getCustomerById",
                description="Retrieve customer from the json data based on id. Fetch customer by id.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"id": {"type": "string", "description": "Customer ID"}},
                    "required": ["id"],
                },
            ),
            types.FunctionDeclaration(
                name="getOrders",
                description="Retrieve all orders from the json data based on limit. Fetch all orders.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "description": "Maximum number of orders"}},
                },
            ),
            types.FunctionDeclaration(
                name="getOrdersWithCustomerDetails",
                description="Retrieve the latest orders along with customer details (name) with optional limit.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "description": "Maximum number of orders"}},
                },
            ),
            types.FunctionDeclaration(
                name="getOrderById",
                description="Retrieve order from the json data based on id. Fetch order by id.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"id": {"type": "string", "description": "Order ID"}},
                    "required": ["id"],
                },
            ),
        ])
    ]


# OpenAI용 tools 배열 (chat.completions)
def openai_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "getWeatherData",
                "description": "Retrieve the live weather data of a city from external API. Get live weather by city or region.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "country": {"type": "string", "description": "Country name (optional)"},
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "getCustomers",
                "description": "Retrieve all customers from the json data based on limit. Fetch all customers.",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "description": "Maximum number of customers"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "getCustomerById",
                "description": "Retrieve customer from the json data based on id. Fetch customer by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "description": "Customer ID"}},
                    "required": ["id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "getOrders",
                "description": "Retrieve all orders from the json data based on limit. Fetch all orders.",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "description": "Maximum number of orders"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "getOrdersWithCustomerDetails",
                "description": "Retrieve the latest orders along with customer details (name) with optional limit.",
                "parameters": {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "description": "Maximum number of orders"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "getOrderById",
                "description": "Retrieve order from the json data based on id. Fetch order by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "description": "Order ID"}},
                    "required": ["id"],
                },
            },
        },
    ]
