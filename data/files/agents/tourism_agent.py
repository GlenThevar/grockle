import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from pydantic import SecretStr
from datetime import date
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from tools.tourism_tools import (
    search_tripadvisor_attractions,
    search_tripadvisor_things_to_do,
    search_tripadvisor_cruises,
    search_tourist_attractions_google_maps,
)

load_dotenv("../env/.env")


def create_llm():

    return ChatBedrockConverse(
        model="qwen.qwen3-next-80b-a3b",
        region_name="us-east-1",
        aws_access_key_id=SecretStr(os.getenv("AWS_ACCESS_KEY_ID") or ""),
        aws_secret_access_key=SecretStr(os.getenv("AWS_SECRET_ACCESS_KEY") or ""),
    )


tourism_agent = create_llm()
tourism_tools = [
    search_tripadvisor_attractions,
    search_tripadvisor_things_to_do,
    search_tripadvisor_cruises,
    search_tourist_attractions_google_maps,
]
tourism_tool_map = {tool.name: tool for tool in tourism_tools}
tourism_agent_with_tools = tourism_agent.bind_tools(tourism_tools)


def run_tourism_agent(query: str):

    messages = [
        SystemMessage(content=f"""
            You are an expert Tourism Travel Assistant.
            """),
        HumanMessage(content=query),
    ]

    while True:

        response = tourism_agent_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:
            print(
                f"\nInvoking tool: {tool_call['name']} with args: {tool_call['args']}\n"
            )
            tool = tourism_tool_map[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            print(f"\nTool result: {result}\n")
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )
