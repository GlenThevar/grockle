import os
import requests
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv("../env/.env")


@tool
def weather_specialist(query: str) -> str:
    """
    Ask the weather specialist.
    """

    from agents.weather_agent import run_weather_agent

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
def get_historical_weather_prediction_open_meteo(
    city: str,
    date: str,
) -> str:
    """
    Predicts weather for dates beyond the 16-day forecast limit by averaging
    historical weather data from Open-Meteo for the past 5 years.

    Args:
        city: City name (e.g., 'Mumbai')
        date: Future date in 'YYYY-MM-DD' format

    Returns:
        A formatted string with predicted weather metrics and historical context.
    """
    try:

        geo_res = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=30,
        )
        geo_res.raise_for_status()
        results = geo_res.json().get("results", [])

        if not results:
            return f"Could not find location: {city}"

        lat, lon = results[0]["latitude"], results[0]["longitude"]

        target_dt = datetime.strptime(date, "%Y-%m-%d")
        month_day = target_dt.strftime("%m-%d")
        target_year = target_dt.year

        max_temps, min_temps, precip_sums = [], [], []

        for i in range(1, 5 + 1):
            past_year = target_year - i
            past_date_str = f"{past_year}-{month_day}"

            res = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": past_date_str,
                    "end_date": past_date_str,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "timezone": "auto",
                },
                timeout=30,
            )
            res.raise_for_status()
            daily = res.json().get("daily", {})

            if daily.get("temperature_2m_max"):
                max_temps.append(daily["temperature_2m_max"][0])
                min_temps.append(daily["temperature_2m_min"][0])
                precip_sums.append(daily["precipitation_sum"][0])

        if not max_temps:
            return f"No historical data found for {city} on {date}."

        avg_max = np.mean(max_temps)
        avg_min = np.min(min_temps)
        avg_precip = np.mean(precip_sums)

        rain_likelihood = (
            "High" if avg_precip > 5.0 else ("Moderate" if avg_precip > 1.0 else "Low")
        )

        return f"""
            City: {city}
            Requested Target Date: {date}
            Prediction Status: Estimated (Averaged from historical data over the past 5 years)
            Predicted Max Temperature: {avg_max:.1f}°C
            Predicted Min Temperature: {avg_min:.1f}°C
            Average Precipitation: {avg_precip:.2f} mm ({rain_likelihood} chance of rain)
            """.strip()

    except requests.exceptions.RequestException as e:
        return f"Open-Meteo Archive API error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
