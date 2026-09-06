import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from pydantic import SecretStr
from datetime import date
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from tools.weather_tools import (
    get_weather_information_openweather,
    get_weather_information_open_meteo,
    get_historical_weather_prediction_open_meteo,
    save_openweather_data,
    save_openmeteo_data,
    save_historical_weather_data,
    read_weather_database,
)

load_dotenv("../env/.env")


def create_llm():

    return ChatBedrockConverse(
        model="qwen.qwen3-next-80b-a3b",
        region_name="us-east-1",
        aws_access_key_id=SecretStr(os.getenv("AWS_ACCESS_KEY_ID") or ""),
        aws_secret_access_key=SecretStr(os.getenv("AWS_SECRET_ACCESS_KEY") or ""),
    )


weather_agent = create_llm()
weather_tools = [
    get_weather_information_openweather,
    get_weather_information_open_meteo,
    get_historical_weather_prediction_open_meteo,
    save_openweather_data,
    save_openmeteo_data,
    save_historical_weather_data,
    read_weather_database,
]
weather_tool_map = {tool.name: tool for tool in weather_tools}
weather_agent_with_tools = weather_agent.bind_tools(weather_tools)


def run_weather_agent(query: str):

    messages = [
        SystemMessage(content=f"""
            You are an expert Weather Travel Assistant.

            TODAY'S DATE: {date.today().strftime('%Y-%m-%d')}

            WORKFLOW RULES (Follow in sequence):

            STEP 1: CHECK DATABASE FIRST
            Always start by calling `read_weather_database` to check if weather data already exists for the target city and date.
            - Choose the correct table:
            - Dates 0-5 days from today -> 'WeatherData_OpenWeather'
            - Dates 6-16 days from today -> 'WeatherData_OpenMeteo'
            - Dates >16 days in future -> 'WeatherData_Historical'

            STEP 2: API FALLBACK (Only if database returns no records)
            If `read_weather_database` returns no data, call the appropriate API tool:
            - 0 to 5 days from today -> `get_weather_information_openweather`
            - 6 to 16 days from today -> `get_weather_information_open_meteo`
            - >16 days in future -> `get_historical_weather_prediction_open_meteo`

            STEP 3: SAVE API DATA
            If an API tool was called in Step 2, immediately save that data using the matching save tool:
            - OpenWeather data -> `save_openweather_data`
            - OpenMeteo data -> `save_openmeteo_data`
            - Historical data -> `save_historical_weather_data`

            OUTPUT RULES:
            - Never hallucinate weather data.
            - Summarize results concisely with:
            1. Weather conditions
            2. Temperatures
            3. Tailored travel advice
            """),
        HumanMessage(content=query),
    ]

    while True:

        response = weather_agent_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:
            print(
                f"\nInvoking tool: {tool_call['name']} with args: {tool_call['args']}\n"
            )
            tool = weather_tool_map[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            print(f"\nTool result: {result}\n")
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
