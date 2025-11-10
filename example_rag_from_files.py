#!/usr/bin/env python3
"""
Example: Reading documents from files for RAG

This shows how to extend example_rag.py to read actual files
instead of using hardcoded sample documents.
"""

from example_rag import DocumentChat
from langchain.schema import Document
import os


def load_text_files(directory):
    """Load all .txt files from a directory."""
    documents = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            documents.append(Document(
                page_content=content,
                metadata={"source": filename}
            ))

    return documents


def load_markdown_files(directory):
    """Load all .md files from a directory."""
    documents = []

    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            documents.append(Document(
                page_content=content,
                metadata={"source": filename, "type": "markdown"}
            ))

    return documents


def main():
    print("=" * 60)
    print("RAG Example - Chat with Your Files")
    print("=" * 60)

    # Initialize
    doc_chat = DocumentChat(model_name="llama3.2")

    # Option 1: Read text files from a directory
    docs_dir = "./my_documents"  # Change this to your directory

    if os.path.exists(docs_dir):
        print(f"\nLoading documents from {docs_dir}...")
        documents = load_text_files(docs_dir)

        if documents:
            doc_chat.add_documents(documents)
            print(f"✓ Loaded {len(documents)} documents")
        else:
            print("No .txt files found!")
            return
    else:
        print(f"Directory {docs_dir} not found!")
        print("Creating sample documents...")

        # Create sample directory and files
        os.makedirs(docs_dir, exist_ok=True)

        sample_files = {
            "langchain.txt": "LangChain is a framework for building LLM applications...",
            "ollama.txt": "Ollama allows running LLMs locally...",
        }

        for filename, content in sample_files.items():
            with open(os.path.join(docs_dir, filename), 'w') as f:
                f.write(content)

        print(f"✓ Created sample files in {docs_dir}")
        print("Run this script again to load them.")
        return

    # Interactive Q&A
    print("\nYou can now ask questions about your documents!")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit']:
            break

        result = doc_chat.ask(question)

        print(f"\nAnswer: {result['answer']}\n")
        print("Sources:")
        for i, doc in enumerate(result['sources'], 1):
            print(f"{i}. {doc.metadata.get('source', 'unknown')}")
        print()


if __name__ == "__main__":
    main()
