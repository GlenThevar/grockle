import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from pydantic import SecretStr
from datetime import date as dt
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from tools.flight_tools import (
    get_flight_information_layovers_serpapi,
    get_flight_information_nolayovers_serpapi,
    get_flight_information_bookingdotcom,
)

load_dotenv("../env/.env")


def create_llm():

    return ChatBedrockConverse(
        model="qwen.qwen3-next-80b-a3b",
        region_name="us-east-1",
        aws_access_key_id=SecretStr(os.getenv("AWS_ACCESS_KEY_ID") or ""),
        aws_secret_access_key=SecretStr(os.getenv("AWS_SECRET_ACCESS_KEY") or ""),
    )


flight_agent = create_llm()
flight_tools = [
    get_flight_information_layovers_serpapi,
    get_flight_information_nolayovers_serpapi,
    get_flight_information_bookingdotcom,
]
flight_tool_map = {tool.name: tool for tool in flight_tools}
flight_agent_with_tools = flight_agent.bind_tools(flight_tools)


def run_flight_agent(query: str):

    messages = [
        SystemMessage(content="""
            You are a Flight Search Assistant. Your primary goal is to fetch multiple flight options from both Booking.com and Google Flights (SerpApi) so the user can compare and choose the best one.

                1. When the user requests DIRECT / NON-STOP flights:
                - Call `get_flight_information_bookingdotcom` with stops="0".
                - Call `get_flight_information_nolayovers_serpapi`.

                2. When the user requests CONNECTING flights OR has NO PREFERENCE:
                - Call `get_flight_information_bookingdotcom` with stops="none" (or "1"/"2").
                - Call `get_flight_information_layovers_serpapi`.

                3. Always call BOTH relevant tools in parallel to provide a comprehensive list.
            """),
        HumanMessage(content=query),
    ]

    while True:

        response = flight_agent_with_tools.invoke(messages)

        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:

            print(f"\nTool call: {tool_call}")

            tool = flight_tool_map[tool_call["name"]]

            result = tool.invoke(tool_call["args"])

            print(f"\nTool result: {result}")

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
