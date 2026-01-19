# MindLoop

A basic Python application demonstrating LangChain integration with Ollama for local LLM inference.

## Features

- Interactive chat interface with conversation memory
- Uses LangChain for prompt engineering and chain management
- Runs completely locally using Ollama (no API keys required)
- Simple and extensible architecture

## Quick Start

### 1. Install Ollama

```bash
# Download from https://ollama.ai
# Or on macOS/Linux:
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull a Model

```bash
ollama pull llama3.2
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 4. Run the App

```bash
python MindLoopChat.py
```

For detailed installation instructions, troubleshooting, and optional dependencies, see **[INSTALL.md](INSTALL.md)**.

## Installation Options

- **Core only** (recommended): `pip install -r requirements.txt`
  - Runs: app.py, example_tools.py, example_chains.py, example_memory_output.py

- **Core + RAG**: `pip install -r requirements.txt -r requirements-rag.txt`
  - Adds: example_rag.py (document search)

- **Everything**: `pip install -r requirements-full.txt`
  - All features and examples

## Usage

Run the application:

```bash
python app.py
```

### Available Commands

- Type your message to chat with the AI
- Type `history` to view conversation history
- Type `quit` or `exit` to end the session

### Example Session

```
You: What is the capital of France?

AI: The capital of France is Paris.

You: What is it famous for?

AI: Paris is famous for many things, including the Eiffel Tower, the Louvre Museum,
Notre-Dame Cathedral, French cuisine, fashion, and art...
```

## Project Structure

```
mindloop/
├── app.py                          # Main chat application
├── example_groq.py                 # Cloud API example (Groq)
├── example_rag.py                  # RAG with sample documents *
├── example_rag_from_files.py       # RAG with your text/markdown files *
├── example_rag_advanced_loaders.py # RAG with PDFs, web pages, etc. *
├── example_tools.py                # Function calling (calculator, search)
├── example_chains.py               # Advanced chain composition
├── example_memory_output.py       # Memory types & output parsers
├── requirements.txt                # Core dependencies
├── requirements-rag.txt            # RAG dependencies *
├── requirements-full.txt           # All dependencies
├── README.md                       # This file
├── INSTALL.md                      # Detailed installation guide
├── EXTENSIBILITY.md                # Comprehensive extensibility guide
└── .gitignore                     # Git ignore rules

* Requires additional dependencies (see INSTALL.md)
```

## How It Works

The application uses:

1. **LangChain** - Framework for building LLM applications
2. **Ollama** - Run large language models locally
3. **ConversationBufferMemory** - Maintain chat history
4. **ChatPromptTemplate** - Structure prompts consistently
5. **LCEL (LangChain Expression Language)** - Chain components together

## Customization

### Change the Model

Edit the model name in `app.py`:

```python
chat = MindLoopChat(model_name="mistral")  # Instead of "llama2"
```

### Modify System Prompt

Edit the system message in `app.py:41`:

```python
("system", "You are a helpful AI assistant. Be concise and helpful."),
```

### Add Streaming

For streaming responses, modify the chain to use callbacks (see LangChain docs).

## Extensibility

This app demonstrates the basic building blocks, but LangChain is highly extensible. See **[EXTENSIBILITY.md](EXTENSIBILITY.md)** for a comprehensive guide on how to extend this app.

### Quick Extension Examples

**1. Add RAG (Chat with Documents)**
```bash
python example_rag.py
```
Demonstrates RAG with hardcoded sample documents. Great for understanding how it works.

Want to use your own files?
```bash
python example_rag_from_files.py        # Read .txt or .md files from a directory
python example_rag_advanced_loaders.py  # Read PDFs, web pages, GitHub repos, etc.
```

**2. Add Tools (Function Calling)**
```bash
python example_tools.py
```
Give the LLM capabilities like calculator, web search, Wikipedia lookup, etc.

**3. Advanced Chains**
```bash
python example_chains.py
```
Build multi-step workflows: sequential, parallel, routing, and transformation chains.

**4. Memory & Output Parsing**
```bash
python example_memory_output.py
```
Different memory types and structured output (JSON, lists, type-safe objects).

### Common Extension Patterns

- **Swap LLM providers**: Easy switch between Ollama, Groq, GPT, Claude
- **Add RAG**: Search your documents, PDFs, websites
- **Add tools**: Calculator, APIs, databases, file operations
- **Chain composition**: Multi-step workflows, parallel processing
- **Memory management**: Buffer, window, summary memory
- **Output parsing**: Get structured JSON instead of text
- **Agents**: Let LLM decide which tools to use

See [EXTENSIBILITY.md](EXTENSIBILITY.md) for detailed guides, code examples, and best practices.

## Alternative Free APIs

While Ollama is great for local development, here are some free API alternatives:

### 1. **Groq** (Recommended for Speed)
- Very fast inference
- Free tier available
- Models: Llama 3, Mixtral, Gemma
- Setup: `pip install langchain-groq`

### 2. **Google Gemini**
- Free tier with generous limits
- Multimodal capabilities
- Setup: `pip install langchain-google-genai`

### 3. **Anthropic Claude** (Limited Free Trial)
- High-quality responses
- Free trial credits
- Setup: `pip install langchain-anthropic`

### 4. **Hugging Face Inference API**
- Access to many open-source models
- Free tier available
- Setup: `pip install langchain-huggingface`

### 5. **Together AI**
- Multiple open-source models
- Free credits on signup
- Fast inference

## Why Use Ollama?

**Pros:**
- 100% local and private
- No API keys or registration
- No rate limits
- Works offline
- Free to use

**Cons:**
- Requires local GPU/CPU resources
- Slower than cloud APIs (depending on hardware)
- Models are smaller than GPT-4 or Claude

## Troubleshooting

### "Error connecting to Ollama"

1. Make sure Ollama is running: `ollama serve`
2. Check if the model is installed: `ollama list`
3. Verify the URL is correct (default: http://localhost:11434)

### "Model not found"

Pull the model first: `ollama pull llama2`

### Slow responses

- Try a smaller model like `phi` or `tinyllama`
- Check your system resources
- Consider using a cloud API for faster responses

## Next Steps

Ready to extend the app? Here's a suggested learning path:

1. **Run the basic app**: `python app.py` - Get familiar with the interface
2. **Try the examples**: Run each example file to see different patterns
3. **Read the guide**: Check out [EXTENSIBILITY.md](EXTENSIBILITY.md) for detailed explanations
4. **Modify examples**: Change prompts, models, or add your own data
5. **Build your use case**: Combine patterns to create your application

### Ideas for Your Next Project

- Personal knowledge assistant (RAG + memory)
- Code review bot (RAG on codebase + tools for running tests)
- Research assistant (RAG + web search + summarization chains)
- Customer support bot (RAG + tools + structured output)
- Content generator (sequential chains + different memory types)
- Create a web UI with Streamlit or Gradio

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Ollama Model Library](https://ollama.ai/library)

## License

See LICENSE file for details.
