#!/usr/bin/env python3
"""
Advanced Chains Example

This demonstrates different types of chains and how to compose them:
- Sequential chains (output of one becomes input to another)
- Routing chains (choose different paths based on input)
- Parallel chains (run multiple chains simultaneously)
- Custom chains with specific logic

Use cases:
- Multi-step workflows
- Content generation pipelines
- Data processing with LLM reasoning
"""

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from typing import Dict


class ChainExamples:
    """Demonstrates various chain patterns."""

    def __init__(self, model_name="llama3.2"):
        self.llm = Ollama(model=model_name, temperature=0.7)

    def sequential_chain_example(self):
        """
        Sequential Chain: Output of one chain feeds into the next.

        Example: Generate a story title -> Write story -> Summarize story
        """
        print("\n=== Sequential Chain Example ===")
        print("Task: Generate story -> Summarize -> Extract moral\n")

        # Chain 1: Generate a short story
        story_template = """Write a very short story (2-3 sentences) about {topic}.

Story:"""
        story_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(template=story_template, input_variables=["topic"]),
            output_key="story"
        )

        # Chain 2: Summarize in one sentence
        summary_template = """Summarize this story in one sentence:

{story}

Summary:"""
        summary_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(template=summary_template, input_variables=["story"]),
            output_key="summary"
        )

        # Chain 3: Extract moral
        moral_template = """What is the moral or lesson from this story?

{story}

Moral:"""
        moral_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(template=moral_template, input_variables=["story"]),
            output_key="moral"
        )

        # Combine into sequential chain
        overall_chain = SequentialChain(
            chains=[story_chain, summary_chain, moral_chain],
            input_variables=["topic"],
            output_variables=["story", "summary", "moral"],
            verbose=True
        )

        # Run the chain
        result = overall_chain.invoke({"topic": "a robot learning to feel emotions"})

        print(f"\nStory:\n{result['story']}")
        print(f"\nSummary:\n{result['summary']}")
        print(f"\nMoral:\n{result['moral']}")

    def parallel_chain_example(self):
        """
        Parallel Chains: Run multiple chains and combine results.

        Example: Analyze sentiment + Extract keywords + Summarize (all at once)
        """
        print("\n=== Parallel Chain Example ===")
        print("Task: Analyze text from multiple perspectives simultaneously\n")

        text = """
        The new restaurant opened last week and I was excited to try it.
        The food was absolutely delicious - especially the pasta dish.
        However, the service was quite slow and the prices were a bit high.
        Overall, I'd recommend it for special occasions but not regular dining.
        """

        # Create three different analysis chains
        sentiment_prompt = PromptTemplate(
            template="What is the sentiment (positive/negative/mixed) of this review? Answer in one word.\n\n{text}\n\nSentiment:",
            input_variables=["text"]
        )

        keywords_prompt = PromptTemplate(
            template="Extract 3 key topics from this review as a comma-separated list.\n\n{text}\n\nKeywords:",
            input_variables=["text"]
        )

        rating_prompt = PromptTemplate(
            template="Based on this review, what rating out of 5 stars would you give? Just say the number.\n\n{text}\n\nRating:",
            input_variables=["text"]
        )

        # Using LCEL for parallel execution
        from langchain.schema.runnable import RunnableParallel

        parallel_chain = RunnableParallel(
            sentiment=sentiment_prompt | self.llm | StrOutputParser(),
            keywords=keywords_prompt | self.llm | StrOutputParser(),
            rating=rating_prompt | self.llm | StrOutputParser()
        )

        result = parallel_chain.invoke({"text": text})

        print(f"Text analyzed:\n{text}")
        print(f"\nResults:")
        print(f"  Sentiment: {result['sentiment'].strip()}")
        print(f"  Keywords: {result['keywords'].strip()}")
        print(f"  Rating: {result['rating'].strip()}/5")

    def routing_chain_example(self):
        """
        Routing Chain: Choose different processing based on input.

        Example: Route question to different specialized chains
        """
        print("\n=== Routing Chain Example ===")
        print("Task: Route questions to specialized handlers\n")

        def route_question(question: str) -> str:
            """Determine which type of question this is."""
            question_lower = question.lower()

            if any(word in question_lower for word in ['calculate', 'math', 'number', '+', '-', '*', '/']):
                return "math"
            elif any(word in question_lower for word in ['code', 'program', 'function', 'python']):
                return "code"
            else:
                return "general"

        # Different prompts for different question types
        math_prompt = PromptTemplate(
            template="Solve this math problem step by step:\n\n{question}\n\nSolution:",
            input_variables=["question"]
        )

        code_prompt = PromptTemplate(
            template="Answer this coding question with a code example:\n\n{question}\n\nAnswer with code:",
            input_variables=["question"]
        )

        general_prompt = PromptTemplate(
            template="Answer this question concisely:\n\n{question}\n\nAnswer:",
            input_variables=["question"]
        )

        # Create routing logic using LCEL
        def route_and_process(inputs: Dict) -> str:
            question = inputs["question"]
            route = route_question(question)

            print(f"Routing to: {route} handler")

            if route == "math":
                chain = math_prompt | self.llm | StrOutputParser()
            elif route == "code":
                chain = code_prompt | self.llm | StrOutputParser()
            else:
                chain = general_prompt | self.llm | StrOutputParser()

            return chain.invoke({"question": question})

        routing_chain = RunnableLambda(route_and_process)

        # Test different question types
        questions = [
            "What is 15 * 234?",
            "How do I write a function to reverse a string in Python?",
            "What is the capital of France?"
        ]

        for q in questions:
            print(f"\nQuestion: {q}")
            answer = routing_chain.invoke({"question": q})
            print(f"Answer: {answer.strip()}\n")
            print("-" * 60)

    def transformation_chain_example(self):
        """
        Transformation Chain: Apply data transformations within the chain.

        Example: Preprocess input -> LLM -> Postprocess output
        """
        print("\n=== Transformation Chain Example ===")
        print("Task: Clean input -> Process -> Format output\n")

        def preprocess(inputs: Dict) -> Dict:
            """Clean and prepare the input."""
            text = inputs["raw_text"]
            # Remove extra whitespace, lowercase
            cleaned = " ".join(text.split()).strip()
            print(f"Preprocessed: '{cleaned}'")
            return {"text": cleaned}

        def postprocess(output: str) -> str:
            """Format the output."""
            # Capitalize, add formatting
            formatted = output.strip().upper()
            return f">>> {formatted} <<<"

        prompt = PromptTemplate(
            template="Translate this to French: {text}\n\nFrench:",
            input_variables=["text"]
        )

        # Chain with transformations
        chain = (
            RunnableLambda(preprocess)
            | prompt
            | self.llm
            | StrOutputParser()
            | RunnableLambda(postprocess)
        )

        raw_input = "   hello    world   this  is   a    test   "
        result = chain.invoke({"raw_text": raw_input})

        print(f"\nOriginal: '{raw_input}'")
        print(f"Result: {result}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Advanced Chains Examples")
    print("=" * 60)

    examples = ChainExamples(model_name="llama3.2")

    print("\nThese examples show different ways to combine LLM calls:")
    print("1. Sequential - One after another")
    print("2. Parallel - Multiple at once")
    print("3. Routing - Different paths based on input")
    print("4. Transformation - With data processing\n")

    # Run examples
    try:
        examples.sequential_chain_example()
        input("\nPress Enter to continue to next example...")

        examples.parallel_chain_example()
        input("\nPress Enter to continue to next example...")

        examples.routing_chain_example()
        input("\nPress Enter to continue to next example...")

        examples.transformation_chain_example()

    except KeyboardInterrupt:
        print("\n\nExamples stopped.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
