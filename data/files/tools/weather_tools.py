import os
import requests
import numpy as np
import sqlite3
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv("../env/.env")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "agentData.db")


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


@tool
def save_openweather_data(
    region_name: str,
    date: str,
    temperature: float,
    feels_like: float,
    humidity: float,
) -> str:
    """Saves OpenWeather data into the WeatherData_OpenWeather database table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO WeatherData_OpenWeather (regionName, date, temperature, feels_like, humidity)
                VALUES (?, ?, ?, ?, ?)
            """,
                (region_name, date, temperature, feels_like, humidity),
            )
        return f"Successfully saved OpenWeather data for {region_name} on {date}."
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def save_openmeteo_data(
    region_name: str,
    date: str,
    max_temperature: float,
    min_temperature: float,
    chance_of_rain: float,
) -> str:
    """Saves Open-Meteo forecast data into the WeatherData_OpenMeteo database table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO WeatherData_OpenMeteo (regionName, date, max_temperature, min_temperature, chance_of_rain)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    region_name,
                    date,
                    max_temperature,
                    min_temperature,
                    chance_of_rain,
                ),
            )
        return f"Successfully saved Open-Meteo data for {region_name} on {date}."
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def save_historical_weather_data(
    region_name: str,
    date: str,
    predicted_max_temperature: float,
    predicted_min_temperature: float,
    average_precipitation: float,
) -> str:
    """Saves historical prediction weather data into the WeatherData_Historical database table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO WeatherData_Historical (regionName, date, predicted_max_temperature, predicted_min_temperature, average_precipitation)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    region_name,
                    date,
                    predicted_max_temperature,
                    predicted_min_temperature,
                    average_precipitation,
                ),
            )
        return f"Successfully saved historical weather prediction for {region_name} on {date}."
    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def read_weather_database(
    table_name: str,
    region_name: Optional[str] = None,
    date: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Reads weather records from the database.

    Args:
        table_name: One of 'WeatherData_OpenWeather',
          'WeatherData_OpenMeteo', or 'WeatherData_Historical'.
        region_name: Filter by city/region name (optional).
        date: Filter by date YYYY-MM-DD (optional).
        limit: Maximum number of rows to return (default 10).
    """
    allowed_tables = {
        "WeatherData_OpenWeather",
        "WeatherData_OpenMeteo",
        "WeatherData_Historical",
    }
    if table_name not in allowed_tables:
        return f"Invalid table name. Choose from: {', '.join(allowed_tables)}"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []

            if region_name:
                query += " AND regionName LIKE ?"
                params.append(f"%{region_name}%")
            if date:
                query += " AND date = ?"
                params.append(date)

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return f"No records found in {table_name} for the given filters."
            results = [dict(row) for row in rows]
            return str(results)

    except Exception as e:
        return f"Database read error: {str(e)}"
