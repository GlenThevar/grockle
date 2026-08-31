import chainlit as cl
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage

from agents import llm_with_tools, tool_map


@cl.on_chat_start
async def start():
    cl.user_session.set(
        "messages",
        [SystemMessage(content="""
                You are 'grockle' a travel agent.
                
                your responsibility is to help users plan their whole travel rigth from the start to the end. 
                
                You have only one tool for the time being which is for getting flight data. So only help users with that for the time being, if they ask anything else be honest and dont make things up.
                
                you have to be consise and to the point. Do not use emojis in the response.
                """)],
    )
    await cl.Message(content="Hey its grockle, how can I help you?").send()


@cl.on_message
async def main(message: cl.Message):
    try:

        messages = cl.user_session.get("messages")
        if messages is None:
            messages = [SystemMessage(content="""
                        You are 'grockle' a travel agent.
                            
                        your responsibility is to help users plan their whole travel rigth from the start to the end. 
                            
                        You have only one tool for the time being which is for getting flight data. So only help users with that for the time being, if they ask anything else be honest and dont make things up.
                        
                        you have to be consise and to the point. Do not use emojis in the response.
                    """)]

        messages.append(HumanMessage(content=message.content))  # type: ignore

        while True:
            response = llm_with_tools.invoke(messages)
            messages.append(response)  # type: ignore

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:

                tool = tool_map[tool_call["name"]]
                result = tool.invoke(tool_call["args"])
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
