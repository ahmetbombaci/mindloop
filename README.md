# MindLoop

A basic Python application demonstrating LangChain integration with Ollama for local LLM inference.

## Features

- Interactive chat interface with conversation memory
- Uses LangChain for prompt engineering and chain management
- Runs completely locally using Ollama (no API keys required)
- Simple and extensible architecture

## Prerequisites

1. **Python 3.8+**
2. **Ollama** - Download and install from [https://ollama.ai](https://ollama.ai)

## Setup

### 1. Install Ollama

```bash
# On macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai
```

### 2. Pull a Model

```bash
# Pull the llama2 model (recommended for beginners)
ollama pull llama2

# Or try other models:
# ollama pull mistral
# ollama pull codellama
# ollama pull phi
```

### 3. Start Ollama Server

```bash
# Ollama usually starts automatically, but you can ensure it's running:
ollama serve
```

### 4. Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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
├── app.py              # Main application with MindLoopChat class
├── example_groq.py     # Example using Groq API (cloud alternative)
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore rules
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

- Add RAG (Retrieval-Augmented Generation) with vector stores
- Implement streaming responses
- Add web search capabilities
- Create a web UI with Streamlit or Gradio
- Add function calling/tools

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/docs)
- [Ollama Model Library](https://ollama.ai/library)

## License

See LICENSE file for details.
