from http import client
import os
import requests
import serpapi
from dotenv import load_dotenv
from langchain_core.tools import tool
from datetime import date

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
    This tool allows you to get flight information using the Duffel API. However it is test data and not real data. So only use this tool when the other tools are not available / working. Use this as only a last resort

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
    Get weather forecast for a city and date using Open Weather.

    This tool can only be used for:
    - Current weather
    - Forecasts up to only 5 days ahead from the current date

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


@tool
def get_weather_information_open_meteo(
    city: str,
    date: str,
) -> str:
    """
    Get weather forecast for a city and date using Open-Meteo.

     This tool can only be used for:
        - Current weather
        - Forecasts up to 16 days ahead from the current date

    Args:
        city: City name (e.g. Mumbai)
        date: Date in YYYY-MM-DD format

    Returns:
        Weather information.
    """

    try:

        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
            },
            timeout=30,
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        results = geo_data.get("results", [])

        if not results:
            return f"Could not find location: {city}"

        latitude = results[0]["latitude"]
        longitude = results[0]["longitude"]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": (
                    "weathercode,"
                    "temperature_2m_max,"
                    "temperature_2m_min,"
                    "precipitation_probability_max"
                ),
                "timezone": "auto",
                "forecast_days": 16,
            },
            timeout=30,
        )

        weather_response.raise_for_status()
        weather_data = weather_response.json()
        dates = weather_data["daily"]["time"]

        if date not in dates:
            print("No data")
            return (
                f"No forecast available for {date}. "
                f"Open-Meteo typically provides forecasts for about 16 days."
            )

        index = dates.index(date)

        max_temp = weather_data["daily"]["temperature_2m_max"][index]
        min_temp = weather_data["daily"]["temperature_2m_min"][index]
        rain_probability = weather_data["daily"]["precipitation_probability_max"][index]
        weather_code = weather_data["daily"]["weathercode"][index]

        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
        }

        weather_description = weather_codes.get(
            weather_code,
            f"Weather code {weather_code}",
        )

        return f"""
            City: {city}
            Date: {date}
            Weather: {weather_description}
            Max Temperature: {max_temp}°C
            Min Temperature: {min_temp}°C
            Chance of Rain: {rain_probability}%
            """.strip()

    except requests.exceptions.RequestException as e:
        return f"Open-Meteo API error: {str(e)}"

    except Exception as e:
        return f"Unexpected error: {str(e)}"


@tool
def get_flight_information_nolayovers_serpapi(
    departure_id: str,
    arrival_id: str,
    currency: str = "INR",
    outbound_date: str = date.today().strftime("%Y-%m-%d"),
    adults: int = 1,
    children: int = 0,
    sort_by: int = 2,
) -> str:
    """
    This tool uses SerpAPI to retrieve one-way flight (which do not have layovers) data from Google Flights. Only use this when you want to get direct flights. If the user wants flights with layovers, use the other tool.

    Args:
        departure_id: The departure airport code (e.g., "BOM" for Mumbai)

        arrival_id: The arrival airport code (e.g., "DEL" for Delhi)

        currency: The currency code for the flight prices (default is "INR")

        outbound_date: The date of departure in "YYYY-MM-DD" format (default is today's date)

        adults: The number of adult passengers (default is 1)

        children: The number of child passengers (default is 0)

        sort_by: The sorting criterion for the search (default is 2). 1 stands for Top flights, 2 stands for Price, 3 stands for Departure time, 4 stands for Arrival time, 5 stands for Duration and 6 stands for Emissions

    Returns:
        A string containing the flight information retrieved from Google Flights.
    """

    api_key = os.getenv("SERPAPI_API_KEY")

    if not api_key:
        return "SERPAPI_API_KEY not found."

    client = serpapi.Client(
        api_key="00ce48528020f1eb062b0768e61f592005507fe471dca034c9eae6bd492c770e"
    )

    try:

        response = client.search(
            {
                "engine": "google_flights",
                "departure_id": departure_id,
                "arrival_id": arrival_id,
                "outbound_date": outbound_date,
                "currency": currency,
                "hl": "en",
                "adults": adults,
                "children": children,
                "sort_by": sort_by,
                "type": 2,
                "stops": 1,
            }
        )

        offers = response["other_flights"]

        if not offers:
            return "No flight offers found."

        results = []

        for offer in offers:

            departure = offer["flights"][0]["departure_airport"]["name"]
            departure_time = offer["flights"][0]["departure_airport"]["time"]
            arrival = offer["flights"][0]["arrival_airport"]["name"]
            arrival_time = offer["flights"][0]["arrival_airport"]["time"]
            airline = offer["flights"][0]["airline"]
            price = offer["price"]
            ticket_currency = currency
            duration = offer["total_duration"]

            results.append(f"""
                Airline: {airline}
                Price: {price} {currency}
                Departure: {departure}
                Departure Time: {departure_time}
                Arrival: {arrival}
                Arrival Time: {arrival_time}
                Duration: {duration}
                Ticket Currency: {ticket_currency}
                """.strip())

        return "\n\n".join(results)

    except Exception as e:
        return f"Serp API error: {str(e)}"


@tool
def get_flight_information_layovers_serpapi(
    departure_id: str,
    arrival_id: str,
    currency: str = "INR",
    outbound_date: str = date.today().strftime("%Y-%m-%d"),
    adults: int = 1,
    children: int = 0,
    sort_by: int = 2,
) -> list[dict]:
    """
    This tool uses SerpAPI to retrieve one-way flight (which have layovers) data from Google Flights. Only use this when you want to get flights with layovers. If the user wants direct flights, use the other tool.

    Args:
        departure_id: Origin airport IATA code.
        arrival_id: Destination airport IATA code.
        currency: Currency for pricing.
        outbound_date: Departure date (YYYY-MM-DD).
        adults: Number of adults.
        children: Number of children.
        sort_by:
            1 = Top flights
            2 = Price
            3 = Departure time
            4 = Arrival time
            5 = Duration
            6 = Emissions

    Returns:
        List of flight itinerary dictionaries.
    """

    api_key = os.getenv("SERP_API_KEY")

    if not api_key:
        return [{"error": "SERPAPI_API_KEY not found."}]

    client = serpapi.Client(api_key=api_key)

    try:

        response = client.search(
            {
                "engine": "google_flights",
                "departure_id": departure_id,
                "arrival_id": arrival_id,
                "outbound_date": outbound_date,
                "currency": currency,
                "hl": "en",
                "adults": adults,
                "children": children,
                "sort_by": sort_by,
                "type": 2,
                "stops": 2,
            }
        )

        offers = response.get("other_flights", [])

        if not offers:
            return []

        results = []

        for offer in offers:

            itinerary = {
                "price": offer.get("price"),
                "currency": currency,
                "trip_type": offer.get("type"),
                "total_duration_minutes": offer.get("total_duration"),
                "stops": len(offer.get("layovers", [])),
                "layovers": [],
                "segments": [],
                "carbon_emissions": offer.get("carbon_emissions", {}),
            }

            for layover in offer.get("layovers", []):

                itinerary["layovers"].append(
                    {
                        "airport": layover.get("name"),
                        "airport_code": layover.get("id"),
                        "duration_minutes": layover.get("duration"),
                    }
                )

            for flight in offer.get("flights", []):

                itinerary["segments"].append(
                    {
                        "airline": flight.get("airline"),
                        "flight_number": flight.get("flight_number"),
                        "aircraft": flight.get("airplane"),
                        "travel_class": flight.get("travel_class"),
                        "departure_airport": flight.get("departure_airport", {}).get(
                            "name"
                        ),
                        "departure_airport_code": flight.get(
                            "departure_airport", {}
                        ).get("id"),
                        "departure_time": flight.get("departure_airport", {}).get(
                            "time"
                        ),
                        "arrival_airport": flight.get("arrival_airport", {}).get(
                            "name"
                        ),
                        "arrival_airport_code": flight.get("arrival_airport", {}).get(
                            "id"
                        ),
                        "arrival_time": flight.get("arrival_airport", {}).get("time"),
                        "duration_minutes": flight.get("duration"),
                        "legroom": flight.get("legroom"),
                        "ticket_also_sold_by": flight.get("ticket_also_sold_by", []),
                        "overnight": flight.get("overnight", False),
                        "often_delayed": flight.get(
                            "often_delayed_by_over_30_min", False
                        ),
                        "extensions": flight.get("extensions", []),
                    }
                )

            results.append(itinerary)

        return results

    except Exception as e:
        return [{"error": f"Serp API error: {str(e)}"}]
