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
]
weather_tool_map = {tool.name: tool for tool in weather_tools}
weather_agent_with_tools = weather_agent.bind_tools(weather_tools)


def run_weather_agent(query: str):

    messages = [
        SystemMessage(content=f"""
            You are an expert Weather Travel Assistant.

            TODAY'S DATE: {date.today().strftime('%Y-%m-%d')}

            TOOL SELECTION RULES:
            1. For dates within 0 to 5 days from today:
            - Call `get_weather_information_openweather`.

            2. For dates within 6 to 16 days from today:
            - Call `get_weather_information_open_meteo`.

            3. For dates more than 16 days in the future:
            - Call `get_historical_weather_prediction_open_meteo`.

            OUTPUT RULES:
            - Never hallucinate weather data; rely strictly on tool responses.
            - If data is available, summarize concisely with:
            1. Weather conditions
            2. Temperatures (Max/Min or Feels Like)
            3. Tailored travel advice (e.g., carrying an umbrella, packing heavy coats).
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
