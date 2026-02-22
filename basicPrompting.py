"""
Python script for adding numbers using an ReAct agent, 
for the Hacker Space lecture.

Author: Henning Thomsen
"""

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Load environment variables from .env file
# Be sure to have valid API keys in this file
load_dotenv()

# Define LLM models
llm_openai = ChatOpenAI(model="gpt-4o", temperature=0.1, verbose=True)
llm_claude = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.1, verbose=True)

# Our prompt
prompt = """Explain in one sentence why one should study Generative AI in Cybersecurity."""

# Run models
response_openai = llm_openai.invoke(prompt)
print("OpenAI:", response_openai)

response_claude = llm_claude.invoke(prompt)
print("Claude:", response_claude)