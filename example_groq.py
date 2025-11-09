#!/usr/bin/env python3
"""
Example using LangChain with Groq API (free tier available).
Groq provides very fast inference with models like Llama 3.

Setup:
1. Sign up at https://console.groq.com
2. Get your API key
3. pip install langchain-groq
4. Set GROQ_API_KEY environment variable or create .env file
"""

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found!")
        print("Please set it in your .env file or environment variables")
        print("\nGet your API key from: https://console.groq.com")
        return

    print("Initializing Groq with Llama 3...")

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama3-70b-8192",  # Fast and powerful
        temperature=0.7,
        api_key=api_key
    )

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("human", "{input}")
    ])

    # Create the chain
    chain = (
        {"input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✓ Connected to Groq successfully!\n")

    # Interactive chat loop
    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break

        try:
            print("\nAI: ", end="", flush=True)
            response = chain.invoke(user_input)
            print(response)
            print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
