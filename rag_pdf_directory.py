#!/usr/bin/env python3
"""
RAG for PDF Documents

This script loads all PDF files from a specified directory,
indexes them in a vector database, and allows you to ask questions.

Setup:
    pip install -r requirements.txt -r requirements-rag.txt
    pip install pypdf  # For PDF support

Usage:
    python rag_pdf_directory.py
"""

# Check for required dependencies
try:
    import chromadb
    import sentence_transformers
except ImportError:
    print("=" * 60)
    print("Missing Required Dependencies")
    print("=" * 60)
    print("\nInstall with:")
    print("  pip install -r requirements-rag.txt")
    print("\nOr manually:")
    print("  pip install chromadb sentence-transformers")
    print("\n" + "=" * 60)
    exit(1)

try:
    import pypdf
except ImportError:
    print("=" * 60)
    print("Missing PDF Support")
    print("=" * 60)
    print("\nInstall with:")
    print("  pip install pypdf")
    print("\n" + "=" * 60)
    exit(1)

from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import glob


class PDFChatbot:
    """Chat with your PDF documents using RAG."""

    def __init__(self, pdf_directory, model_name="llama3.2"):
        """
        Initialize the PDF chatbot.

        Args:
            pdf_directory: Path to directory containing PDF files
            model_name: Ollama model to use (default: llama3.2)
        """
        self.pdf_directory = pdf_directory
        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None

        print("=" * 60)
        print("PDF RAG Chatbot")
        print("=" * 60)

        # Initialize LLM
        print(f"\nInitializing Ollama with {model_name}...")
        self.llm = Ollama(model=model_name)

        # Initialize embeddings
        print("Loading embeddings model (this may take a moment)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    def load_pdfs(self):
        """Load all PDF files from the directory."""
        print(f"\nSearching for PDFs in: {self.pdf_directory}")

        # Check if directory exists
        if not os.path.exists(self.pdf_directory):
            print(f"\n✗ Error: Directory '{self.pdf_directory}' does not exist!")
            return False

        # Find all PDF files
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))

        if not pdf_files:
            print(f"\n✗ No PDF files found in '{self.pdf_directory}'")
            return False

        print(f"✓ Found {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files:
            print(f"  - {os.path.basename(pdf)}")

        # Load all PDFs
        print("\nLoading PDFs...")
        all_documents = []

        for pdf_file in pdf_files:
            try:
                print(f"  Loading {os.path.basename(pdf_file)}...", end=" ")
                loader = PyPDFLoader(pdf_file)
                documents = loader.load()
                all_documents.extend(documents)
                print(f"✓ ({len(documents)} pages)")
            except Exception as e:
                print(f"✗ Error: {e}")

        if not all_documents:
            print("\n✗ No documents were loaded successfully!")
            return False

        print(f"\n✓ Total pages loaded: {len(all_documents)}")

        # Split documents into chunks
        print("\nSplitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        splits = text_splitter.split_documents(all_documents)
        print(f"✓ Created {len(splits)} text chunks")

        # Create vector store
        print("\nCreating vector database...")
        print("(This may take a few minutes for the first run)")

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_pdf_db"
        )

        print("✓ Vector database created!")

        # Create QA chain
        template = """Use the following context from the PDF documents to answer the question.
If you cannot find the answer in the provided context, say "I cannot find this information in the PDFs."

Context from PDFs:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        print("\n" + "=" * 60)
        print("✓ Ready to answer questions!")
        print("=" * 60)

        return True

    def ask(self, question):
        """Ask a question about the PDF documents."""
        if self.qa_chain is None:
            return {
                "answer": "Please load PDFs first!",
                "sources": []
            }

        print("\nSearching PDFs...")
        result = self.qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

    def interactive_chat(self):
        """Start an interactive Q&A session."""
        print("\nYou can now ask questions about your PDFs!")
        print("\nCommands:")
        print("  - Type your question to get an answer")
        print("  - Type 'sources' to see the last sources used")
        print("  - Type 'pdfs' to list loaded PDF files")
        print("  - Type 'quit' or 'exit' to end")
        print("\n" + "-" * 60)

        last_sources = []

        while True:
            try:
                question = input("\nQuestion: ").strip()

                if not question:
                    continue

                if question.lower() in ['quit', 'exit']:
                    print("\nGoodbye!")
                    break

                if question.lower() == 'pdfs':
                    pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))
                    print("\nLoaded PDFs:")
                    for pdf in pdf_files:
                        print(f"  - {os.path.basename(pdf)}")
                    continue

                if question.lower() == 'sources':
                    if last_sources:
                        print("\nLast sources used:")
                        for i, doc in enumerate(last_sources, 1):
                            source = doc.metadata.get('source', 'unknown')
                            page = doc.metadata.get('page', '?')
                            print(f"\n{i}. {os.path.basename(source)} (page {page + 1})")
                            print(f"   {doc.page_content[:150]}...")
                    else:
                        print("\nNo previous sources available.")
                    continue

                # Get answer
                result = self.ask(question)
                last_sources = result['sources']

                print(f"\nAnswer:\n{result['answer']}")

                # Show sources
                if result['sources']:
                    print("\nSources:")
                    sources_seen = set()
                    for doc in result['sources']:
                        source = doc.metadata.get('source', 'unknown')
                        page = doc.metadata.get('page', '?')
                        source_key = f"{source}:{page}"

                        if source_key not in sources_seen:
                            sources_seen.add(source_key)
                            print(f"  • {os.path.basename(source)} (page {page + 1})")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}")


def main():
    """Main function."""
    # Configuration
    PDF_DIRECTORY = "/Users/ahmetbombaci/rag/"
    MODEL_NAME = "llama3.2"

    # You can change these settings:
    # PDF_DIRECTORY = input("Enter PDF directory path: ").strip()
    # MODEL_NAME = input("Enter model name (default llama3.2): ").strip() or "llama3.2"

    # Initialize chatbot
    chatbot = PDFChatbot(
        pdf_directory=PDF_DIRECTORY,
        model_name=MODEL_NAME
    )

    # Load PDFs
    if not chatbot.load_pdfs():
        print("\nFailed to load PDFs. Please check the directory and try again.")
        return

    # Start interactive chat
    chatbot.interactive_chat()


if __name__ == "__main__":
    main()
