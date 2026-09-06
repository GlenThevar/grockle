import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from pydantic import SecretStr
from datetime import date
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from tools.flight_tools import (
    get_flight_information_layovers_serpapi,
    get_flight_information_nolayovers_serpapi,
    get_flight_information_bookingdotcom,
    save_flight_direct_serpapi,
    save_flight_layovers_serpapi,
    save_flight_bookingdotcom,
    read_flight_database,
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
    save_flight_direct_serpapi,
    save_flight_layovers_serpapi,
    save_flight_bookingdotcom,
    read_flight_database,
]
flight_tool_map = {tool.name: tool for tool in flight_tools}
flight_agent_with_tools = flight_agent.bind_tools(flight_tools)


def run_flight_agent(query: str):

    messages = [
        SystemMessage(content=f"""
            You are an expert Flight Search Assistant.
            TODAY'S DATE: {date.today().strftime('%Y-%m-%d')}

            WORKFLOW RULES (Follow in sequence):

            STEP 1: CHECK DATABASE FIRST
                Always start by calling `read_flight_database` to check for cached flight deals.
                    - For Direct queries: Check `FlightData_SerpApi_Direct` or `FlightData_BookingDotCom`.
                        - If data is present for any of the two queries ( tools ) then dont go to step 2 and 3 
                    - For Layover queries: Check `FlightData_SerpApi_Layovers` and `FlightData_BookingDotCom`.
                        - - If data is present for any of the two queries ( tools ) then dont go to step 2 and 3  

            STEP 2: API FALLBACK (Only if database returns no records)
                1. DIRECT / NON-STOP Flights (ONLY CALL ONE OF THE TWO TOOLS, NOT BOTH. IF ONE RETURNS NO RESULT THEN YOU CAN CALL THE OTHER ):
                    - Call `get_flight_information_bookingdotcom` (stops="0")
                    - Call `get_flight_information_nolayovers_serpapi`

                2. CONNECTING / NO PREFERENCE Flight (ONLY CALL ONE OF THE TWO TOOLS, NOT BOTH. IF ONE RETURNS NO RESULT THEN YOU CAN CALL THE OTHER):
                    - Call `get_flight_information_bookingdotcom` (stops="none")
                    - Call `get_flight_information_layovers_serpapi`

            STEP 3: SAVE API DATA
                If API tools were called in Step 2, immediately save ALL THE results to SQLite ( SAVE ALL THE RESULTS RECEIVED ):
                    - Direct SerpAPI -> `save_flight_direct_serpapi`
                    - Layover SerpAPI -> `save_flight_layovers_serpapi`
                    - Booking.com -> `save_flight_bookingdotcom`

            OUTPUT RULES:
                - Present a clear comparison table (Airlines, Prices, Times, Duration).
            """),
        HumanMessage(content=query),
    ]

    while True:

        response = flight_agent_with_tools.invoke(messages)

        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:

            print(
                f"\nInvoking tool: {tool_call['name']} with args: {tool_call['args']}\n"
            )

            tool = flight_tool_map[tool_call["name"]]

            result = tool.invoke(tool_call["args"])

            print(f"\nTool result: {result}\n")

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
