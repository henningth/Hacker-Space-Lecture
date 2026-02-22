"""
Python script for adding numbers using an ReAct agent, 
for the Hacker Space lecture.

Author: Henning Thomsen
"""

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, AIMessage

from langchain.tools import tool
from langchain.agents import create_agent


# Load environment variables from .env file
# Be sure to have valid API keys in this file
load_dotenv()

# Define system prompt
system_prompt = """You are a helpful assistant"""

# Define LLMs
# OpenAI model
OPENAI_MODEL = "gpt-4o-mini"
openai_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1)

# Open source model via Groq
GROQ_MODEL = "llama3-8b-8192"
groq_llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)

# Define the addition tool
@tool
def add_numbers(inputs: str) -> str:
    """
    Use this tool to add numbers. Input should be a space-separated list of numbers.
    """
    numbers = [float(x) for x in inputs.split() if x.replace('.', '', 1).isdigit()]
    return str(sum(numbers))

tools = [add_numbers]

# Create the agent with our tool and llm
agent = create_agent(
    model=openai_llm, # try other models also
    tools=tools,
    system_prompt=system_prompt # try other variations also
)

QUERY = f"What is 5.2 plus 3.8?"

response = agent.invoke({"messages": [HumanMessage(QUERY)]})

print("Agent Response:", response) # Full response object, below we extract the steps

for message in response["messages"]:
    if type(message).__name__ == "AIMessage" and len(message.content) == 0:
        print(type(message).__name__ + ": " + str(message.tool_calls) + "\n")
    elif type(message).__name__ == "AIMessage" and len(message.content) != 0:
        print(type(message).__name__ + ": " + message.content + "\n")
    else:
        print(type(message).__name__ + ": " + message.content + "\n")