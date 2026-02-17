import os
import json
import urllib.request
import urllib.parse


def get_weather_data(city: str, country: str | None = None) -> dict:
    """Retrieve the live weather data of a city from external API.
    Get live weather by city or region.
    """
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return {"error": "WEATHER_API_KEY is not set"}
    query = f"{city}, {country}" if country else city
    url = (
        "https://api.weatherapi.com/v1/current.json?key="
        + api_key
        + "&q="
        + urllib.parse.quote(query)
        + "&aqi=no"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}
