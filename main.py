import chainlit as cl
from langchain_aws import ChatBedrockConverse
import os
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

llm = ChatBedrockConverse(
    model="qwen.qwen3-next-80b-a3b",
    region_name="us-east-1"
)

@cl.on_chat_start
async def start():
    await cl.Message(content="Hello! How can I help you").send()
    
    
    
@cl.on_message
async def main(message: cl.Message):
    try:
        response = await llm.ainvoke(message.content)
        await cl.Message(content=str(response.content)).send()
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()   
        