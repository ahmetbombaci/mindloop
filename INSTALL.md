# Installation Guide

This guide will help you install MindLoop and its dependencies correctly.

## Quick Start (Recommended)

### 1. Install Ollama

First, install Ollama to run local LLMs:

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai
```

### 2. Pull a Model

```bash
# Pull llama3.2 (recommended, 2GB)
ollama pull llama3.2

# Or try other models:
ollama pull llama2      # 3.8GB
ollama pull mistral     # 4.1GB
ollama pull phi         # 1.6GB - faster, smaller
```

### 3. Install Python Dependencies

Choose the installation option that matches what you want to run:

#### Option A: Core Only (Recommended to Start)

Install just the core dependencies for basic chat and tools examples:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

**What you can run:**
- `MindLoopChat.py` - Basic chat with memory
- `example_groq.py` - Cloud API example (with API key)
- `MindLoopTools.py` - Function calling (calculator, Wikipedia, etc.)
- `example_chains.py` - Advanced chain composition
- `example_memory_output.py` - Memory and output parsing

#### Option B: Core + RAG

Add RAG (document search) capabilities:

```bash
pip install -r requirements.txt -r requirements-rag.txt
```

**Additional capabilities:**
- `example_rag.py` - Chat with documents using vector search

#### Option C: Everything

Install all dependencies including optional ones:

```bash
pip install -r requirements-full.txt
```

## Testing Your Installation

### Test Basic Chat

```bash
python app.py
```

You should see:
```
============================================================
MindLoop - LangChain + Ollama Chat Application
============================================================
✓ Connected to Ollama successfully!
```

### Test Each Example

```bash
# Tools example (should work with core install)
python example_tools.py

# RAG example (requires RAG dependencies)
python example_rag.py

# Chains example (works with core install)
python example_chains.py

# Memory example (works with core install)
python example_memory_output.py
```

## Troubleshooting

### Python Version Issues

**Problem:** Installation fails with errors about building wheels or Rust compiler

**Solution:** This project works best with Python 3.8-3.12. If you're using Python 3.13+, some packages may not have pre-built wheels yet.

```bash
# Check your Python version
python3 --version

# If 3.13+, consider using Python 3.11 or 3.12
# Use pyenv to manage multiple Python versions
pyenv install 3.12
pyenv local 3.12
```

### Missing Dependencies

**Problem:** Error like `ModuleNotFoundError: No module named 'sentence_transformers'`

**Solution:** The examples check for required dependencies and will tell you what to install:

```bash
# For RAG example
pip install -r requirements-rag.txt

# Or manually
pip install chromadb sentence-transformers
```

### Ollama Connection Errors

**Problem:** `Error connecting to Ollama`

**Solution:**

1. Make sure Ollama is running:
```bash
ollama serve
```

2. Check if the model is installed:
```bash
ollama list
```

3. If model is missing, pull it:
```bash
ollama pull llama3.2
```

4. Verify Ollama is accessible:
```bash
curl http://localhost:11434/api/tags
```

### Slow Installation

**Problem:** `pip install` is very slow or hangs

**Solutions:**

1. Use a faster mirror:
```bash
pip install -r requirements.txt -i https://pypi.org/simple
```

2. Install without cache:
```bash
pip install -r requirements.txt --no-cache-dir
```

3. Install packages one by one to identify the slow one:
```bash
pip install langchain
pip install langchain-community
pip install ollama
# etc.
```

### ChromaDB Issues

**Problem:** Errors installing or running ChromaDB

**Solution:** ChromaDB has some system dependencies:

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install cmake build-essential

# Then reinstall
pip install --upgrade chromadb
```

### Sentence Transformers Download

**Problem:** First run of RAG example takes a long time

**Explanation:** The first time you run the RAG example, it downloads the embedding model (~80MB). This is normal and only happens once.

**Solution:** Be patient, or pre-download:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

## Requirements Files Explained

The project has three requirements files:

### requirements.txt (Core)
```
langchain==0.1.0
langchain-community==0.0.10
ollama==0.1.6
python-dotenv==1.0.0
requests==2.31.0
```

**Size:** ~50MB
**Use for:** Basic chat, tools, chains, memory examples

### requirements-rag.txt (RAG Add-on)
```
chromadb==0.4.22
sentence-transformers==2.3.1
```

**Size:** ~500MB (includes ML models)
**Use for:** Document search with embeddings

### requirements-full.txt (Everything)
Combines core + RAG + optional provider packages

**Size:** ~500MB
**Use for:** All examples, all features

## Platform-Specific Notes

### macOS

- Apple Silicon (M1/M2/M3): Everything should work great. Ollama uses Metal for GPU acceleration.
- Intel Macs: Works fine, but slower LLM inference.

### Linux

- Ubuntu/Debian: Fully supported
- May need to install build tools for some packages:
  ```bash
  sudo apt-get update
  sudo apt-get install build-essential python3-dev
  ```

### Windows

- Use PowerShell or Command Prompt
- Activate venv with: `venv\Scripts\activate`
- Ollama runs in WSL2 or natively on Windows

## Alternative: Docker (Advanced)

If you prefer Docker:

```dockerfile
# Create a Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "app.py"]
```

```bash
docker build -t mindloop .
docker run -it --network host mindloop
```

**Note:** Ollama must still be running on your host machine.

## Cloud API Setup (Optional)

If you want to use cloud APIs instead of local Ollama:

### Groq (Fast & Free)

1. Sign up at https://console.groq.com
2. Get your API key
3. Create `.env` file:
```
GROQ_API_KEY=your_key_here
```
4. Install:
```bash
pip install langchain-groq
```
5. Run:
```bash
python example_groq.py
```

### Other Providers

See commented lines in `requirements-full.txt`:
```bash
# Uncomment the provider you want to use
pip install langchain-google-genai    # Google Gemini
pip install langchain-anthropic       # Claude
pip install langchain-openai          # OpenAI GPT
```

## Getting Help

1. Check error messages - they usually tell you what's missing
2. Make sure Ollama is running: `ollama serve`
3. Verify model is installed: `ollama list`
4. Check Python version: `python3 --version` (3.8-3.12 recommended)
5. Try installing dependencies one at a time
6. Use a fresh virtual environment if all else fails

## Next Steps

Once installed successfully:

1. Run `python app.py` to try basic chat
2. Run the example files to see different features
3. Read [EXTENSIBILITY.md](EXTENSIBILITY.md) to learn how to extend
4. Build your own application!
