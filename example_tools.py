#!/usr/bin/env python3
"""
Tools/Function Calling Example

This demonstrates how to give your LLM access to custom tools/functions.
The LLM can use these tools to perform actions beyond text generation.

Use cases:
- Calculator for math
- Web search for current information
- API calls to external services
- Database queries
- File operations
"""

from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import math
import datetime
import requests


# Define custom tools
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "sqrt(16)"

    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation with math functions
        result = eval(expression, {"__builtins__": {}}, {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        })
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {e}"


def get_current_time(query: str) -> str:
    """
    Returns the current date and time.

    Args:
        query: Any query about current time (ignored, kept for consistency)

    Returns:
        Current date and time
    """
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def search_wikipedia(query: str) -> str:
    """
    Searches Wikipedia for information.

    Args:
        query: Search term

    Returns:
        Summary from Wikipedia
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return data.get('extract', 'No summary available')
        else:
            return f"Could not find information about '{query}'"
    except Exception as e:
        return f"Error searching Wikipedia: {e}"


def word_counter(text: str) -> str:
    """
    Counts words, characters, and sentences in text.

    Args:
        text: The text to analyze

    Returns:
        Statistics about the text
    """
    words = len(text.split())
    chars = len(text)
    sentences = text.count('.') + text.count('!') + text.count('?')

    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"


class ToolsChat:
    """Chat with access to custom tools."""

    def __init__(self, model_name="llama3.2"):
        print("Initializing chat with tools...")

        # Initialize LLM
        self.llm = Ollama(model=model_name, temperature=0)

        # Define tools
        self.tools = [
            Tool(
                name="Calculator",
                func=calculator,
                description="Useful for mathematical calculations. Input should be a valid mathematical expression like '2+2' or 'sqrt(16)'."
            ),
            Tool(
                name="CurrentTime",
                func=get_current_time,
                description="Useful for getting the current date and time. Input can be any question about current time."
            ),
            Tool(
                name="Wikipedia",
                func=search_wikipedia,
                description="Useful for searching information on Wikipedia. Input should be a search term or topic."
            ),
            Tool(
                name="TextAnalyzer",
                func=word_counter,
                description="Useful for analyzing text and counting words, characters, and sentences. Input should be the text to analyze."
            )
        ]

        # Initialize agent
        # Note: For Ollama, we use ZERO_SHOT_REACT_DESCRIPTION which works better with local models
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,  # Shows the agent's thinking process
            max_iterations=3,
            early_stopping_method="generate"
        )

        print("✓ Tools loaded successfully!")
        print("\nAvailable tools:")
        for tool in self.tools:
            print(f"  - {tool.name}: {tool.description}")

    def chat(self, message: str) -> str:
        """Send a message and get a response."""
        try:
            response = self.agent.invoke({"input": message})
            return response["output"]
        except Exception as e:
            return f"Error: {e}"


def main():
    """Example usage."""
    print("=" * 60)
    print("Tools Example - Chat with Function Calling")
    print("=" * 60)
    print("\nThe AI can now use tools to help answer your questions!")
    print("Try asking it to:")
    print("  - Calculate something: 'What is 15 * 234?'")
    print("  - Get current time: 'What time is it?'")
    print("  - Search Wikipedia: 'Tell me about Python programming'")
    print("  - Analyze text: 'Count words in: Hello world example'")
    print("\nType 'quit' to exit\n")

    # Initialize
    chat = ToolsChat(model_name="llama3.2")

    # Interactive chat
    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit']:
            break

        print("\nAI: ", end="", flush=True)
        response = chat.chat(user_input)
        print(response)


if __name__ == "__main__":
    main()
