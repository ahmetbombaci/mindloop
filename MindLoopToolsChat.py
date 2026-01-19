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
from cli import parse_args, DEFAULT_MODEL
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
        Extended content from Wikipedia
    """
    # Wikipedia API requires a User-Agent header
    headers = {"User-Agent": "MindLoop/1.0 (https://github.com/mindloop; educational project)"}

    def get_extract(title: str) -> str:
        """Get extended extract using MediaWiki API."""
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": False,  # Get more than just intro
            "explaintext": True,  # Plain text, no HTML
            "exsectionformat": "plain",
            "exchars": 2000,  # Get up to 2000 characters
            "format": "json"
        }
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if "extract" in page:
                    return page["extract"]
        return None

    try:
        # First, search for the page title
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1
        }
        search_response = requests.get(search_url, headers=headers, params=params, timeout=5)

        if search_response.status_code == 200:
            search_data = search_response.json()
            results = search_data.get("query", {}).get("search", [])
            if results:
                page_title = results[0]["title"]
                extract = get_extract(page_title)
                if extract:
                    return f"Wikipedia article '{page_title}':\n\n{extract}"

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


class MindLoopToolsChat:
    """Chat with access to custom tools."""

    def __init__(self, model_name=DEFAULT_MODEL):
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
                description="Useful for fetching Wikipedia article content. Input should be a topic name. Returns article text that you should then summarize for the user."
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
            max_iterations=5,
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
    args = parse_args()
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
    chat = MindLoopToolsChat(model_name=args.model)

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
