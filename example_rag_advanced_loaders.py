#!/usr/bin/env python3
"""
Advanced Document Loading for RAG

Shows how to use LangChain's built-in document loaders to read:
- PDFs
- Web pages
- CSV files
- Code repositories
- And more

Install additional dependencies:
    pip install pypdf  # For PDF support
    pip install beautifulsoup4  # For web scraping
"""

from example_rag import DocumentChat


def load_from_pdf(pdf_path):
    """Load documents from a PDF file."""
    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} pages from PDF")
        return documents
    except ImportError:
        print("Install pypdf: pip install pypdf")
        return []


def load_from_url(url):
    """Load content from a webpage."""
    try:
        from langchain_community.document_loaders import WebBaseLoader

        loader = WebBaseLoader(url)
        documents = loader.load()
        print(f"✓ Loaded content from {url}")
        return documents
    except ImportError:
        print("Install beautifulsoup4: pip install beautifulsoup4")
        return []


def load_from_directory(directory, glob_pattern="**/*.txt"):
    """Load all files matching a pattern from a directory."""
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader

    loader = DirectoryLoader(
        directory,
        glob=glob_pattern,
        loader_cls=TextLoader
    )
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} files from {directory}")
    return documents


def load_from_github(repo_url, branch="main"):
    """Load files from a GitHub repository."""
    try:
        from langchain_community.document_loaders import GitLoader

        # Note: This clones the repo to a temp directory
        loader = GitLoader(
            clone_url=repo_url,
            repo_path="./temp_repo",
            branch=branch,
        )
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} files from GitHub")
        return documents
    except ImportError:
        print("Install gitpython: pip install gitpython")
        return []


def load_from_csv(csv_path):
    """Load data from a CSV file."""
    from langchain_community.document_loaders import CSVLoader

    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} rows from CSV")
    return documents


def main():
    print("=" * 60)
    print("Advanced RAG Document Loading Examples")
    print("=" * 60)

    # Initialize
    doc_chat = DocumentChat(model_name="llama3.2")

    # Choose your document source
    print("\nSelect document source:")
    print("1. PDF file")
    print("2. Web page")
    print("3. Directory of text files")
    print("4. GitHub repository")
    print("5. CSV file")

    choice = input("\nChoice (1-5): ").strip()

    documents = []

    if choice == "1":
        pdf_path = input("Enter PDF path: ").strip()
        documents = load_from_pdf(pdf_path)

    elif choice == "2":
        url = input("Enter URL: ").strip()
        documents = load_from_url(url)

    elif choice == "3":
        directory = input("Enter directory path: ").strip()
        pattern = input("File pattern (default **/*.txt): ").strip() or "**/*.txt"
        documents = load_from_directory(directory, pattern)

    elif choice == "4":
        repo_url = input("Enter GitHub repo URL: ").strip()
        branch = input("Branch (default main): ").strip() or "main"
        documents = load_from_github(repo_url, branch)

    elif choice == "5":
        csv_path = input("Enter CSV path: ").strip()
        documents = load_from_csv(csv_path)

    else:
        print("Invalid choice!")
        return

    if not documents:
        print("No documents loaded!")
        return

    # Add documents to RAG system
    doc_chat.add_documents(documents)

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
            source = doc.metadata.get('source', 'unknown')
            print(f"{i}. {source[:80]}...")
        print()


if __name__ == "__main__":
    main()
