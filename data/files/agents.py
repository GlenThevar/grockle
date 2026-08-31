import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from pydantic import SecretStr
from datetime import date as dt
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from tools import (
    get_flight_information_duffel,
    get_weather_information_openweather,
    flight_specialist,
    weather_specialist,
)

load_dotenv("../env/.env")


def create_llm():

    return ChatBedrockConverse(
        model="qwen.qwen3-next-80b-a3b",
        region_name="us-east-1",
        aws_access_key_id=SecretStr(os.getenv("AWS_ACCESS_KEY_ID") or ""),
        aws_secret_access_key=SecretStr(os.getenv("AWS_SECRET_ACCESS_KEY") or ""),
    )


# Flight Agent
flight_agent = create_llm()
flight_tools = [get_flight_information_duffel]
flight_tool_map = {tool.name: tool for tool in flight_tools}
flight_agent_with_tools = flight_agent.bind_tools(flight_tools)


def run_flight_agent(query: str):

    messages = [
        SystemMessage(content="""
            You are a Flight travel assistant.

            Use available tools whenever needed.
            
            """),
        HumanMessage(content=query),
    ]

    while True:

        response = flight_agent_with_tools.invoke(messages)

        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:

            tool = flight_tool_map[tool_call["name"]]

            result = tool.invoke(tool_call["args"])

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )


# Weather Agent

weather_agent = create_llm()
weather_tools = [get_weather_information_openweather]
weather_tool_map = {tool.name: tool for tool in weather_tools}
weather_agent_with_tools = weather_agent.bind_tools(weather_tools)


def run_weather_agent(query: str):

    messages = [
        SystemMessage(content=f"""
            You are a weather travel assistant.

            Use available tools whenever needed.

            Never make up weather information.

            If weather data is available,
            summarize it in a concise and
            easy-to-understand manner.
            
            todays date is {dt.today() or ""}

            Mention:
            - Weather conditions
            - Temperature
            - Any important travel advice
            """),
        HumanMessage(content=query),
    ]

    while True:

        response = weather_agent_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:
            tool = weather_tool_map[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )


# Main Agent
main_agent = create_llm()
main_tools = [flight_specialist, weather_specialist]
tool_map = {tool.name: tool for tool in main_tools}
llm_with_tools = main_agent.bind_tools(main_tools)
