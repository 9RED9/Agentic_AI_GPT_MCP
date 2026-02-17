"""공통 도구 실행 로직. LLM API와 무관하게 동일한 결과를 반환한다."""
import json
import os
import urllib.request
import urllib.parse


def get_current_weather(location: str) -> dict:
    """지정한 지역의 현재 날씨를 weatherapi.com으로 조회한다."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return {"error": "WEATHER_API_KEY is not set. Add it to .env or environment."}
    url = (
        "https://api.weatherapi.com/v1/current.json?key="
        + api_key
        + "&q="
        + urllib.parse.quote(location)
        + "&aqi=no"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}


def get_all_customers(limit: int | None = None) -> list:
    """등록된 고객 목록을 반환한다. limit이 있으면 해당 개수만 반환한다."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, "customers.json")
    with open(path, "r", encoding="utf-8") as f:
        customers = json.load(f)
    if limit is not None and limit > 0:
        return customers[:limit]
    return customers
