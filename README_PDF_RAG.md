# How to Use RAG with Your PDF Directory

This guide shows you how to chat with PDFs in `/Users/ahmetbombaci/rag/`

## Quick Start

### 1. Install Dependencies

```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install RAG dependencies
pip install -r requirements.txt -r requirements-rag.txt

# Install PDF support
pip install pypdf
```

### 2. Add PDFs to Your Directory

Place your PDF files in:
```
/Users/ahmetbombaci/rag/
```

For example:
```
/Users/ahmetbombaci/rag/
├── document1.pdf
├── research_paper.pdf
├── manual.pdf
└── report.pdf
```

### 3. Run the Script

```bash
python rag_pdf_directory.py
```

## What Happens

1. **Finds PDFs**: Searches for all `.pdf` files in `/Users/ahmetbombaci/rag/`
2. **Loads them**: Extracts text from each PDF page
3. **Splits text**: Breaks documents into chunks (1000 chars each with 200 char overlap)
4. **Creates embeddings**: Converts text to vectors using sentence-transformers
5. **Stores in database**: Saves to ChromaDB in `./chroma_pdf_db` directory
6. **Ready to chat**: You can now ask questions!

## Example Session

```
============================================================
PDF RAG Chatbot
============================================================

Initializing Ollama with llama3.2...
Loading embeddings model (this may take a moment)...

Searching for PDFs in: /Users/ahmetbombaci/rag/
✓ Found 3 PDF file(s):
  - document1.pdf
  - research_paper.pdf
  - manual.pdf

Loading PDFs...
  Loading document1.pdf... ✓ (5 pages)
  Loading research_paper.pdf... ✓ (12 pages)
  Loading manual.pdf... ✓ (8 pages)

✓ Total pages loaded: 25

Splitting documents into chunks...
✓ Created 87 text chunks

Creating vector database...
(This may take a few minutes for the first run)
✓ Vector database created!

============================================================
✓ Ready to answer questions!
============================================================

You can now ask questions about your PDFs!

Commands:
  - Type your question to get an answer
  - Type 'sources' to see the last sources used
  - Type 'pdfs' to list loaded PDF files
  - Type 'quit' or 'exit' to end

------------------------------------------------------------

Question: What is the main topic of the research paper?

Searching PDFs...

Answer:
The main topic of the research paper is machine learning applications
in natural language processing, with a focus on transformer architectures.

Sources:
  • research_paper.pdf (page 1)
  • research_paper.pdf (page 3)

Question: sources

Last sources used:

1. research_paper.pdf (page 1)
   This paper presents a comprehensive survey of transformer-based
   models in NLP. We examine the evolution from the original...

2. research_paper.pdf (page 3)
   The attention mechanism allows models to weigh the importance
   of different parts of the input when making predictions...

Question: quit

Goodbye!
```

## Available Commands

During the chat session, you can use:

- **Ask questions**: Just type your question
- **`sources`**: See the exact PDF pages used for the last answer
- **`pdfs`**: List all loaded PDF files
- **`quit`** or **`exit`**: End the session

## Configuration

If you want to change the directory or model, edit `rag_pdf_directory.py`:

```python
# Line 229-230
PDF_DIRECTORY = "/Users/ahmetbombaci/rag/"  # Change this
MODEL_NAME = "llama3.2"                      # Or use mistral, llama2, etc.
```

Or uncomment lines 233-234 to be prompted for these values each time.

## How It Works

### Text Chunking
- **Chunk size**: 1000 characters
- **Overlap**: 200 characters
- **Why?**: Breaks large PDFs into manageable pieces while preserving context

### Retrieval
- **Top K**: Retrieves 4 most relevant chunks per question
- **Search**: Uses semantic similarity (meaning-based, not keyword-based)

### Database
- **Location**: `./chroma_pdf_db` directory
- **Persistence**: Database is saved, so you don't need to reload PDFs every time
- **Reset**: Delete the `chroma_pdf_db` folder to rebuild from scratch

## Tips

### For Better Answers

1. **Be specific**: "What are the three main findings in section 2?" is better than "What does it say?"
2. **Reference sections**: If your PDFs have section numbers or titles, mention them
3. **Ask follow-ups**: The context from previous answers is not maintained, so include context in your question

### For Large PDF Collections

If you have many PDFs (50+ files or 500+ pages):

1. **Increase chunk retrieval**: Edit line 187 to retrieve more chunks:
   ```python
   search_kwargs={"k": 8}  # Instead of 4
   ```

2. **Use a more powerful model**:
   ```bash
   ollama pull mistral
   ```
   Then change `MODEL_NAME = "mistral"`

3. **Smaller chunks**: Edit line 126-127 for more granular search:
   ```python
   chunk_size=500,
   chunk_overlap=100
   ```

### First Run Takes Longer

The first time you run this:
- Downloads the embedding model (~80MB)
- Processes all PDFs
- Creates the vector database

Subsequent runs will be faster because:
- Embedding model is cached
- You can reuse the database (unless you add new PDFs)

### Adding New PDFs

If you add new PDFs to the directory:

1. Delete the old database:
   ```bash
   rm -rf chroma_pdf_db
   ```

2. Run the script again:
   ```bash
   python rag_pdf_directory.py
   ```

Or modify the script to add an "update" feature.

## Troubleshooting

### "No PDF files found"

Check:
1. PDFs are in `/Users/ahmetbombaci/rag/`
2. Files have `.pdf` extension (not `.PDF` on Linux)
3. Directory path is correct

### "Error loading PDF"

Some PDFs are scanned images, not text. Solutions:
1. Use OCR: Install `pytesseract` and modify the loader
2. Convert to text: Use a PDF-to-text tool first
3. Skip the problematic file

### "Out of memory"

If you have hundreds of large PDFs:
1. Process in batches
2. Use smaller chunk size
3. Increase system RAM
4. Use a cloud-based vector database

### Slow responses

1. Make sure Ollama is using GPU (if available)
2. Try a smaller model: `ollama pull phi`
3. Reduce number of retrieved chunks (edit `k` value)

## Next Steps

Once this works, you can:

1. **Add a web UI**: Use Streamlit or Gradio
2. **Support more formats**: Add Word, Excel, etc.
3. **Multi-directory support**: Load from multiple locations
4. **Conversation memory**: Track chat history
5. **Export answers**: Save Q&A to a file
6. **Advanced search**: Filter by PDF, page range, or date

See [EXTENSIBILITY.md](EXTENSIBILITY.md) for more ideas!
