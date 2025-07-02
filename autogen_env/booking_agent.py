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
from autogen_agentchat.teams import SelectorGroupChat

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

model_client = AzureOpenAIChatCompletionClient(
     model=deployment_model,
     api_key=api_key,
     azure_endpoint=azure_endpoint,
     azure_deployment=deployment_model,
     api_version=api_version
)

# Web Surfer Agent
web_surfer = MultimodalWebSurfer(
     name="web_surfer",
     model_client=model_client,
     headless=True, # Enables visible browser window
     animate_actions=False
 )

travel_planner_agent = AssistantAgent(
    name='travel_planner_agent',
    model_client=model_client,
    # tools=[web_surfer],
    system_message="You are a travel planner. You will counsult and gather the holiday requirements of customers. You will then make a detailed plan for the holiday, including flights, hotels, and activities. You may use web_surfer to find information if need to verify your plan. You will then instruct the flight booking agent and hotel booking agent to find the best options for the customer.",
)

# Flight Booking Agent
flight_booking_agent = AssistantAgent(
    name='flight_booking_agent',
    model_client=model_client,
    # tools=[web_surfer],
    system_message="You are a flight booking agent. Use the web_surfer agent to find flights that meets all the travelling planer's requirement. You don't have to reallybook the flight but play back the information of the flight found, including time, price, airline and link to book the flight. Try https://www.skyscanner.net/, https://uk.trip.com/ and https://www.kayak.co.uk/ first. Always choose the first flight the web_sufer finds. If no flights are found, instruct the WebSurfer agent to try alternative search terms or websites.",
)

hotel_booking_agent = AssistantAgent(
    name='hotel_booking_agent',
    model_client=model_client,
    # tools=[web_surfer],
    system_message="You are a hotel booking agent. Use the web_surfer agent to find hotels that meets all the travelling planer's requirement. You don't have toy book but play back the information of the hotels found, including price, location, rating and link to book the hotel. Try https://www.booking.com/. Always choose the first hotel the web_sufer finds. No worries about the rates, reviwing or stars. If no hotels are found, instruct the WebSurfer agent to try alternative search terms or websites.",
)   

customer_proxy = UserProxyAgent(name="customer")

selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""

team = SelectorGroupChat(
    participants=[customer_proxy, travel_planner_agent, flight_booking_agent, hotel_booking_agent, web_surfer],
    termination_condition=TextMentionTermination("exit", sources=["customer_proxy"]),
    model_client=model_client,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,  
)

async def main():
    # Start the task and stream responses to the terminal
    stream = team.run_stream(
        task="Help the customer plan a holiday"
    )
    
    # The Console streams the agent interactions to your terminal live
    await Console(stream)

    # Close connections after execution
    await web_surfer.close()
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())
    print("🛍️ Shopping Complete: ✅")
