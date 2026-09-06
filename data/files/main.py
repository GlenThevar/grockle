import sqlite3
import os
import chainlit as cl
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from datetime import date

from agents.main_agent import llm_with_tools, tool_map


def init_db(db_name="db/agentData.db"):
    with sqlite3.connect(db_name) as conn:
        os.makedirs(os.path.dirname(db_name), exist_ok=True)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS WeatherData_OpenWeather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_DATE,
                regionName TEXT,
                date TEXT,
                temperature REAL,
                feels_like REAL,
                humidity REAL
            )
            """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS WeatherData_OpenMeteo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_DATE,
                regionName TEXT,
                date TEXT,
                max_temperature REAL,
                min_temperature REAL,
                chance_of_rain REAL
            )
            """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS WeatherData_Historical (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_DATE,
                regionName TEXT,
                date TEXT,
                predicted_max_temperature REAL,
                predicted_min_temperature REAL,
                average_precipitation REAL
            )
            """)


@cl.on_chat_start
async def start():
    init_db()
    cl.user_session.set(
        "messages",
        [SystemMessage(content=f"""
            You are Grockle, an AI travel assistant.

            Your long-term responsibility is to help users plan trips from start to finish.
            
            TODAY'S DATE: {date.today().strftime('%Y-%m-%d')}

            Current Capabilities:
            - Flight search (direct and connecting options).
            - Weather forecasts and historical climate estimates.
            - Do not provide information about hotels, restaurants, attractions, visas, or local transit until relevant tools become available.
            - If asked about unsupported topics, clearly explain your current limitations without inventing answers.

            Required Information Before Tool Execution:

            For Flight Queries:
            1. Departure city or airport
            2. Arrival city or airport
            3. Departure date
            4. Number of adults travelling
            5. Number of children travelling
            6. Flight preference (1=Top Flights, 2=Lowest Price, 3=Earliest Departure, 4=Earliest Arrival, 5=Shortest Duration, 6=Lowest Emissions)
            7. Whether layovers are acceptable

            For Weather Queries:
            1. Target city
            2. Date of travel (format: YYYY-MM-DD)

            Response Guidelines:
            - Be concise, professional, and helpful.
            - Do not use emojis.
            - Do not make assumptions; ask brief follow-up questions if required details are missing.
            - Present flight and weather results clearly, summarizing key metrics.
            """)],
    )
    await cl.Message(content="Hey its grockle, how can I help you?").send()


@cl.on_message
async def main(message: cl.Message):
    try:

        messages = cl.user_session.get("messages")
        if messages is None:
            messages = [SystemMessage(content="""
                You are Grockle, an AI travel assistant.

                Your long-term responsibility is to help users plan trips from start to finish.

                Current Capabilities:
                - Flight search (direct and connecting options).
                - Weather forecasts and historical climate estimates.
                - Do not provide information about hotels, restaurants, attractions, visas, or local transit until relevant tools become available.
                - If asked about unsupported topics, clearly explain your current limitations without inventing answers.

                Required Information Before Tool Execution:

                For Flight Queries:
                1. Departure city or airport
                2. Arrival city or airport
                3. Departure date
                4. Number of adults travelling
                5. Number of children travelling
                6. Flight preference (1=Top Flights, 2=Lowest Price, 3=Earliest Departure, 4=Earliest Arrival, 5=Shortest Duration, 6=Lowest Emissions)
                7. Whether layovers are acceptable

                For Weather Queries:
                1. Target city
                2. Date of travel (format: YYYY-MM-DD)

                Response Guidelines:
                - Be concise, professional, and helpful.
                - Do not use emojis.
                - Do not make assumptions; ask brief follow-up questions if required details are missing.
                - Present flight and weather results clearly, summarizing key metrics.
        """)]

        messages.append(HumanMessage(content=message.content))  # type: ignore

        while True:
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)  # type: ignore

            print(f"\nLLM Response: {response.content}\n")

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                print(
                    f"\nInvoking tool: {tool_call['name']} with args: {tool_call['args']}\n"
                )

                tool = tool_map[tool_call["name"]]
                result = await tool.ainvoke(tool_call["args"])

                print(f"\nTool result: {result}\n")

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )  # type: ignore
                )

        cl.user_session.set("messages", messages)
        await cl.Message(content=str(response.content)).send()
    except Exception as e:
        print(f"Error: {e}")
