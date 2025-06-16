## 🚀 02 Running Agents

Create a script in the project root of your repository e.g ```shopping_agents.py``` and follow these steps:

1. Add your imports:
```python
import asyncio
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
```
2. Load your envrionment variables from your .env file:
```python
load_dotenv()
```
3. Create variables from your .env file:

```python
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
```

4. Define your model:
```python

model_client = AzureOpenAIChatCompletionClient(
     model="gpt-4o",
     api_key=api_key,
     azure_endpoint=azure_endpoint,
     azure_deployment="gpt-4o",
     api_version=api_version
)
```

5. Define your agents:
```python
#Web Surfer Agent
web_surfer = MultimodalWebSurfer(
     name="web_surfer",
     model_client=model_client,
     headless=False,           # Enables visible browser window
     animate_actions=True
 )

#User Proxy Agent
user_proxy = UserProxyAgent(name="user_proxy")
```
6. Create your team with the Round Robin Group Chat and set Termination condition:
```python
team = RoundRobinGroupChat(
    agents=[web_surfer, user_proxy],
    termination_condition=TextMentionTermination("exit", sources=["user_proxy"])
)
```
> The team uses RoundRobinGroupChat to alternate turns between the agents. The session ends when the user_proxy agent receives the word "exit" from your terminal input.

7. Define your stream logic:
    stream = team.run_stream(
        task="Browse the e-commerce site https://www.currys.co.uk/ and add headphones to the shopping basket."
    )
    await Console(stream)

    await web_surfer.close()
    await model_client.close()

Put this all inside a main function and run:
```python

#run the main function

if __name__ == "__main__":
    asyncio.run(main())
    print("Shopping Complete ✅")
```

### 🧩 How It All Comes Together

Once you have configured your agents, run your script in the terminal:

A browser window should pop up, watch how the agent is naviagting the web on its own. You may see a red dot moving around and clicking on the webpage, like a human would!
If you toggle between the web browser and the terminal you can see a break down of what the agents are doing:

1. The **task instruction** tells the team what to do (e.g. “Go to an e-commerce site and add headphones to the cart”)
2. The agents take turns thinking and acting
3. The **WebSurfer** opens the browser and performs the web navigation
4. The **UserProxyAgent** lets you observe or interact if needed in the terminal
5. When done, you type `exit` to stop in the terminal

---

 > ⚠️ There may be times when agents fail to complete a task. Common reasons include:

- Security restrictions (e.g CAPTCHA, login prompts, bot detection)
- Incomplete instructions or vague goals
- Unsupported site layouts or dynamic JavaScript rendering

---

### 💡 Creative Challenges (Optional Tasks)

These extra tasks help you explore AutoGen’s flexibility and push the limits of your web agent:

- Did your agent manage to add items to the basket successfully?  
   If not, examine the terminal logs to understand what went wrong. Try:
  - Rephrasing the task more specifically
  - Asking the assistant to redirect to an alternative site
  - Adding intermediate steps (e.g., “search first, then filter by price”)

- Use the **UserProxyAgent** to interact mid-run:  
  Type guidance, clarifications, or commands directly into the terminal - you’re part of the loop!
<img width="550" alt="image" src="https://github.com/user-attachments/assets/9caa6aa4-93e4-44ad-afbd-9ac138b9739a" />

- Create a **Custom Assistant Agent** with a unique system message.  
  For example, add an agent with this behavior:
  > “Instruct the WebSurfer agent to try alternative product pages if a site returns no results.”

- 🔁 Try another group chat type like:
  - **SelectorGroupChat**: Automatically routes queries to the most relevant agent.
  - **MagenticOne**: Builds structured, tool-enhanced workflows from agent capabilities.

---
Designing reliable multi-agent systems is a real-world engineering challenge. Coordination often breaks down so building effective agentic workflows is an iterative process of trial, observation, and refinement.

Have fun exploring AutoGen! It’s a powerful platform for building smart, collaborative agents that can read, search, click, and reason across the web!
