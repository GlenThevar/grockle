import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv("../env/.env")


@tool
def get_flight_information_duffel(
    origin: str,
    destination: str,
    departure_date: str,
    passenger_type: str = "adult",
    cabin_class: str = "economy",
) -> str:
    """
    Search flights using Duffel.

    Args:
        origin: Origin IATA code (e.g. BOM)
        destination: Destination IATA code (e.g. DEL)
        departure_date: YYYY-MM-DD
        passenger_type: adult, child, infant_without_seat
        cabin_class: economy, premium_economy, business, first

    Returns:
        Flight information as a formatted string.
    """

    api_key = os.getenv("DUFFEL_API_KEY")

    if not api_key:
        return "DUFFEL_API_KEY not found."

    url = "https://api.duffel.com/air/offer_requests"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Duffel-Version": "v2",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "data": {
            "slices": [
                {
                    "origin": origin,
                    "destination": destination,
                    "departure_date": departure_date,
                }
            ],
            "passengers": [
                {
                    "type": passenger_type,
                }
            ],
            "cabin_class": cabin_class,
        }
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        offers = data.get("data", {}).get("offers", [])

        if not offers:
            return "No flight offers found."

        offers = sorted(offers, key=lambda x: float(x["total_amount"]))

        results = []

        for offer in offers[:5]:

            segment = offer["slices"][0]["segments"][0]

            airline = offer["owner"]["name"]

            price = offer["total_amount"]

            currency = offer["total_currency"]

            departure = segment["departing_at"]

            arrival = segment["arriving_at"]

            duration = segment["duration"]

            results.append(f"""
                Airline: {airline}
                Price: {price} {currency}
                Departure: {departure}
                Arrival: {arrival}
                Duration: {duration}
                """.strip())

        return "\n\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"Duffel API error: {str(e)}"

    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def flight_specialist(query: str) -> str:
    """
    Ask the flight specialist.
    """

    from agents import run_flight_agent

    return run_flight_agent(query)


@tool
def weather_specialist(query: str) -> str:
    """
    Ask the weather specialist.
    """

    from agents import run_weather_agent

    return run_weather_agent(query)


@tool
def get_weather_information_openweather(
    city: str,
    date: str,
) -> str:
    """
    Get weather forecast for a city and date.

    This tool can only be used for:
    - Current weather
    - Forecasts up to approximately 5 days ahead

    Args:
        city: City name (e.g. Mumbai)
        date: Date in YYYY-MM-DD format

    Returns:
        Weather information.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "OPENWEATHER_API_KEY not found."

    url = "https://api.openweathermap.org/data/2.5/forecast"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric",
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        forecasts = data.get("list", [])

        if not forecasts:
            return f"No weather forecast found for {city}."

        matching_forecasts = []

        for forecast in forecasts:

            forecast_datetime = forecast["dt_txt"]

            if forecast_datetime.startswith(date):

                matching_forecasts.append(
                    {
                        "time": forecast_datetime,
                        "temperature": forecast["main"]["temp"],
                        "feels_like": forecast["main"]["feels_like"],
                        "humidity": forecast["main"]["humidity"],
                        "weather": forecast["weather"][0]["description"],
                    }
                )

        if not matching_forecasts:
            return (
                f"No forecast available for {date}. "
                f"OpenWeather forecast is limited to approximately 5 days."
            )

        results = []

        for item in matching_forecasts:

            results.append(f"""
                Time: {item['time']}
                Weather: {item['weather']}
                Temperature: {item['temperature']}°C
                Feels Like: {item['feels_like']}°C
                Humidity: {item['humidity']}%
            """.strip())

        return "\n\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"OpenWeather API error: {str(e)}"

    except Exception as e:
        return f"Unexpected error: {str(e)}"
