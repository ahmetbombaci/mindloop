#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Example

This demonstrates how to extend the basic chat app with document search.
The LLM can now answer questions based on your own documents.

Setup:
    pip install -r requirements.txt -r requirements.txt

    Or manually:
    pip install chromadb sentence-transformers

Use case: Chat with your documents, create knowledge bases
"""

# Check for required dependencies
try:
    import chromadb
    import sentence_transformers
except ImportError as e:
    print("=" * 60)
    print("Missing Required Dependencies")
    print("=" * 60)
    print("\nThis example requires additional packages.")
    print("\nInstall with:")
    print("  pip install -r requirements.txt")
    print("\nOr manually:")
    print("  pip install chromadb sentence-transformers")
    print("\n" + "=" * 60)
    exit(1)

from cli import DEFAULT_MODEL, parse_args
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os


class DocumentChat:
    """Chat with your documents using RAG."""

    def __init__(self, model_name="llama3.2"):
        print("Initializing RAG system...")

        # Initialize LLM
        self.llm = Ollama(model=model_name)

        # Initialize embeddings (runs locally, no API needed)
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Small, fast model
            model_kwargs={'device': 'cpu'}
        )

        # Initialize vector store
        self.vectorstore = None
        self.qa_chain = None

    def add_documents(self, texts, metadatas=None):
        """
        Add documents to the knowledge base.

        Args:
            texts: List of text strings or Document objects
            metadatas: Optional list of metadata dicts
        """
        # Split texts into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        # Create documents
        if isinstance(texts[0], str):
            docs = [Document(page_content=text, metadata=metadatas[i] if metadatas else {})
                   for i, text in enumerate(texts)]
        else:
            docs = texts

        # Split documents
        split_docs = text_splitter.split_documents(docs)

        print(f"Adding {len(split_docs)} document chunks...")

        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        else:
            self.vectorstore.add_documents(split_docs)

        # Create QA chain
        template = """Use the following context to answer the question.
If you don't know the answer based on the context, say so.

Context: {context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        print("✓ Documents indexed successfully!")

    def ask(self, question):
        """Ask a question about your documents."""
        if self.qa_chain is None:
            return "No documents loaded. Please add documents first."

        result = self.qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }


def main():
    args = parse_args()
    """Example usage."""
    print("=" * 60)
    print("RAG Example - Chat with Your Documents")
    print("=" * 60)

    # Initialize
    doc_chat = DocumentChat(model_name=DEFAULT_MODEL)

    # Example: Add some documents
    sample_docs = [
        "LangChain is a framework for developing applications powered by language models. "
        "It enables applications that are context-aware and can reason.",

        "Ollama allows you to run open-source LLMs locally. "
        "It supports models like Llama 2, Mistral, and Code Llama.",

        "RAG stands for Retrieval-Augmented Generation. It combines retrieval of relevant "
        "documents with text generation to provide accurate, context-aware answers.",

        "Vector databases store embeddings, which are numerical representations of text. "
        "They enable semantic search, finding documents by meaning rather than keywords."
    ]

    metadatas = [
        {"source": "langchain_docs", "topic": "framework"},
        {"source": "ollama_docs", "topic": "local_llm"},
        {"source": "rag_tutorial", "topic": "rag"},
        {"source": "vector_db_guide", "topic": "embeddings"}
    ]

    doc_chat.add_documents(sample_docs, metadatas)

    # Interactive Q&A
    print("\nYou can now ask questions about the documents!")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit']:
            break

        print("\nSearching documents...\n")
        result = doc_chat.ask(question)

        print(f"Answer: {result['answer']}\n")

        print("Sources used:")
        for i, doc in enumerate(result['sources'], 1):
            print(f"{i}. {doc.metadata.get('topic', 'unknown')} - {doc.page_content[:100]}...")
        print()


if __name__ == "__main__":
    main()
