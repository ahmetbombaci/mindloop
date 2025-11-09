#!/usr/bin/env python3
"""
Basic LangChain application using Ollama for local LLM inference.
This app demonstrates a simple chat interface with conversation memory.
"""

from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
import os


class MindLoopChat:
    """A simple chatbot using LangChain and Ollama."""

    def __init__(self, model_name="llama2", base_url="http://localhost:11434"):
        """
        Initialize the chat application.

        Args:
            model_name: Name of the Ollama model to use (default: llama2)
            base_url: URL of the Ollama server (default: http://localhost:11434)
        """
        print(f"Initializing MindLoop with model: {model_name}")

        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            base_url=base_url
        )

        # Create a conversation memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )

        # Create a prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Be concise and helpful."),
            ("human", "{input}")
        ])

        # Create the chain
        self.chain = (
            {"input": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def chat(self, user_input):
        """
        Send a message and get a response.

        Args:
            user_input: The user's message

        Returns:
            The AI's response
        """
        response = self.chain.invoke(user_input)

        # Store in memory
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)

        return response

    def get_history(self):
        """Get the conversation history."""
        return self.memory.chat_memory.messages


def main():
    """Main function to run the interactive chat."""
    print("=" * 60)
    print("MindLoop - LangChain + Ollama Chat Application")
    print("=" * 60)
    print("\nMake sure Ollama is running locally!")
    print("Install Ollama from: https://ollama.ai")
    print("Then run: ollama pull llama2")
    print("\nType 'quit' or 'exit' to end the conversation")
    print("Type 'history' to see conversation history")
    print("-" * 60)

    # Initialize the chat
    try:
        chat = MindLoopChat(model_name="llama2")
        print("\n✓ Connected to Ollama successfully!\n")
    except Exception as e:
        print(f"\n✗ Error connecting to Ollama: {e}")
        print("Make sure Ollama is running on http://localhost:11434")
        return

    # Interactive chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye! 👋")
                break

            if user_input.lower() == 'history':
                print("\n--- Conversation History ---")
                for msg in chat.get_history():
                    role = "You" if msg.type == "human" else "AI"
                    print(f"{role}: {msg.content}")
                print("--- End of History ---")
                continue

            # Get response
            print("\nAI: ", end="", flush=True)
            response = chat.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()
