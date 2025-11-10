#!/usr/bin/env python3
"""
Memory and Output Parsing Examples

This demonstrates:
1. Different types of memory for conversation management
2. Structured output parsing (JSON, lists, etc.)
3. Custom output formats

Use cases:
- Different conversation patterns
- Extracting structured data from LLM responses
- Building APIs that return JSON
"""

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)
from langchain.chains import LLMChain
from langchain.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
    PydanticOutputParser,
    CommaSeparatedListOutputParser
)
from langchain.schema import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
import json


class MemoryExamples:
    """Demonstrates different memory types."""

    def __init__(self, model_name="llama3.2"):
        self.llm = Ollama(model=model_name, temperature=0.7)

    def buffer_memory_example(self):
        """
        ConversationBufferMemory: Keeps entire conversation history.

        Good for: Short conversations, full context needed
        """
        print("\n=== Buffer Memory (Full History) ===")

        memory = ConversationBufferMemory()

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""The following is a conversation with an AI.

{history}
Human: {input}
AI:"""
        )

        chain = LLMChain(llm=self.llm, prompt=prompt, memory=memory)

        # Have a conversation
        print("Human: My name is Alice")
        response1 = chain.invoke({"input": "My name is Alice"})
        print(f"AI: {response1['text']}\n")

        print("Human: What's the weather like?")
        response2 = chain.invoke({"input": "What's the weather like?"})
        print(f"AI: {response2['text']}\n")

        print("Human: What's my name?")
        response3 = chain.invoke({"input": "What's my name?"})
        print(f"AI: {response3['text']}\n")

        print("Memory contents:")
        print(memory.load_memory_variables({}))

    def window_memory_example(self):
        """
        ConversationBufferWindowMemory: Keeps only last K messages.

        Good for: Long conversations, limited context
        """
        print("\n=== Window Memory (Last 2 Exchanges) ===")

        memory = ConversationBufferWindowMemory(k=2)  # Keep last 2 exchanges

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""Recent conversation:

{history}
Human: {input}
AI:"""
        )

        chain = LLMChain(llm=self.llm, prompt=prompt, memory=memory)

        messages = [
            "My name is Bob",
            "I like pizza",
            "I work as an engineer",
            "What's my name?"  # Should not remember (outside window)
        ]

        for msg in messages:
            print(f"Human: {msg}")
            response = chain.invoke({"input": msg})
            print(f"AI: {response['text']}\n")


class OutputParserExamples:
    """Demonstrates structured output parsing."""

    def __init__(self, model_name="llama3.2"):
        self.llm = Ollama(model=model_name, temperature=0)

    def list_parser_example(self):
        """
        Parse comma-separated list from LLM output.

        Good for: Getting lists of items
        """
        print("\n=== List Output Parser ===")

        parser = CommaSeparatedListOutputParser()

        prompt = PromptTemplate(
            template="List 5 programming languages.\n{format_instructions}\n",
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.llm | parser

        result = chain.invoke({})
        print(f"Result type: {type(result)}")
        print(f"Languages: {result}")
        print(f"First language: {result[0]}")

    def structured_parser_example(self):
        """
        Parse structured data with multiple fields.

        Good for: Extracting specific information
        """
        print("\n=== Structured Output Parser ===")

        # Define the structure we want
        response_schemas = [
            ResponseSchema(name="name", description="The person's name"),
            ResponseSchema(name="age", description="The person's age"),
            ResponseSchema(name="occupation", description="The person's job"),
            ResponseSchema(name="hobbies", description="List of hobbies")
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        prompt = PromptTemplate(
            template="Extract information about the person from this text:\n{text}\n\n{format_instructions}",
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.llm | parser

        text = "John is a 30 year old software engineer who enjoys hiking, reading, and playing guitar."

        result = chain.invoke({"text": text})
        print(f"Result type: {type(result)}")
        print(f"Parsed data: {json.dumps(result, indent=2)}")

    def pydantic_parser_example(self):
        """
        Parse into Pydantic models for type safety.

        Good for: Production apps, type-safe APIs
        """
        print("\n=== Pydantic Output Parser ===")

        # Define a Pydantic model
        class Person(BaseModel):
            name: str = Field(description="The person's full name")
            age: int = Field(description="The person's age in years")
            email: str = Field(description="The person's email address")
            skills: List[str] = Field(description="List of professional skills")

        parser = PydanticOutputParser(pydantic_object=Person)

        prompt = PromptTemplate(
            template="Create a fictional person profile based on: {description}\n\n{format_instructions}",
            input_variables=["description"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | self.llm | parser

        result = chain.invoke({"description": "a senior data scientist"})

        print(f"Result type: {type(result)}")
        print(f"Name: {result.name}")
        print(f"Age: {result.age}")
        print(f"Email: {result.email}")
        print(f"Skills: {', '.join(result.skills)}")
        print(f"\nFull object: {result}")

    def custom_parser_example(self):
        """
        Custom output parser for specific format.

        Good for: Custom data formats
        """
        print("\n=== Custom Output Parser ===")

        def parse_rating(text: str) -> dict:
            """Custom parser for rating format."""
            lines = text.strip().split('\n')
            result = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    result[key.strip().lower()] = value.strip()

            return result

        prompt = PromptTemplate(
            template="""Rate this product on a scale of 1-5 for the following categories.
Format each line as "Category: rating"

Product: {product}

Ratings:""",
            input_variables=["product"]
        )

        from langchain.schema.runnable import RunnableLambda

        chain = prompt | self.llm | StrOutputParser() | RunnableLambda(parse_rating)

        result = chain.invoke({"product": "wireless headphones"})

        print(f"Ratings: {json.dumps(result, indent=2)}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Memory and Output Parsing Examples")
    print("=" * 60)

    # Memory examples
    memory_ex = MemoryExamples(model_name="llama3.2")

    try:
        memory_ex.buffer_memory_example()
        input("\nPress Enter to continue...")

        memory_ex.window_memory_example()
        input("\nPress Enter to continue...")

        # Output parser examples
        parser_ex = OutputParserExamples(model_name="llama3.2")

        parser_ex.list_parser_example()
        input("\nPress Enter to continue...")

        parser_ex.structured_parser_example()
        input("\nPress Enter to continue...")

        # Note: Pydantic parser might not work well with local models
        # Uncomment to try:
        # parser_ex.pydantic_parser_example()
        # input("\nPress Enter to continue...")

        parser_ex.custom_parser_example()

        print("\n✓ All examples completed!")

    except KeyboardInterrupt:
        print("\n\nExamples stopped.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
