import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

model_client = AzureOpenAIChatCompletionClient(
     model="gpt-4o",
     api_key=api_key,
     azure_endpoint=azure_endpoint,
     azure_deployment="gpt-4o",
     api_version=api_version
)

#Web Surfer Agent
web_surfer = MultimodalWebSurfer(
     name="web_surfer",
     model_client=model_client,
     headless=False,           # Enables visible browser window
     animate_actions=True
 )

async def web_search(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."

assistantAgent = AssistantAgent(
    name="assistant",
    model_client=model_client,
    tools=[web_search],
    system_message="Instruct the WebSurfer agent to try alternative product pages if a site returns no results.",
)

#User Proxy Agent
user_proxy = UserProxyAgent(name="user_proxy")

team = RoundRobinGroupChat(
    participants=[web_surfer, assistantAgent],
    termination_condition=TextMentionTermination("exit", sources=["user_proxy"])
)

async def main():
    # Start the task and stream responses to the terminal
    stream = team.run_stream(
        task="Browse the e-commerce site https://www.amazon.co.uk/ and add headphones to the shopping basket. Add the first item to the basket."
    )
    
    # The Console streams the agent interactions to your terminal live
    await Console(stream)

    # Close connections after execution
    await web_surfer.close()
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
    print("🛍️ Shopping Complete: ✅")