import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from pydantic import SecretStr

from tools.flight_tools import flight_specialist
from tools.weather_tools import weather_specialist

load_dotenv("../env/.env")


def create_llm():
    return ChatBedrockConverse(
        model="qwen.qwen3-next-80b-a3b",
        region_name="us-east-1",
        aws_access_key_id=SecretStr(os.getenv("AWS_ACCESS_KEY_ID") or ""),
        aws_secret_access_key=SecretStr(os.getenv("AWS_SECRET_ACCESS_KEY") or ""),
    )


main_agent = create_llm()
main_tools = [flight_specialist, weather_specialist]
tool_map = {tool.name: tool for tool in main_tools}
llm_with_tools = main_agent.bind_tools(main_tools)
