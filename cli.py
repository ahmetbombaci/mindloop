import argparse

# Default configuration
DEFAULT_MODEL = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434"

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MindLoop - LangChain + Ollama Chat")
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--base-url", "-u",
        default=DEFAULT_BASE_URL,
        help=f"Ollama server URL (default: {DEFAULT_BASE_URL})"
    )
    return parser.parse_args()