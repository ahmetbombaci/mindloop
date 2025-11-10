# LangChain Extensibility Guide

This guide explains how the MindLoop app can be extended using LangChain's modular architecture.

> **Note:** Different examples require different dependencies. See [INSTALL.md](INSTALL.md) for installation options. The core examples work with just `requirements.txt`, while RAG examples need `requirements-rag.txt`.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Extension Points](#extension-points)
3. [Examples Included](#examples-included)
4. [Common Patterns](#common-patterns)
5. [Best Practices](#best-practices)

## Core Concepts

LangChain is built on a few key abstractions that make it highly extensible:

### 1. **Components** (Building Blocks)
- **LLMs/Chat Models**: The language model (Ollama, GPT, Claude, etc.)
- **Prompts**: Templates for structuring input
- **Memory**: Conversation history management
- **Tools**: Functions the LLM can call
- **Retrievers**: Document search systems
- **Output Parsers**: Structure LLM responses

### 2. **Chains** (Compositions)
- **LLMChain**: Simple prompt + LLM
- **Sequential Chain**: A → B → C
- **Parallel Chain**: A, B, C (all at once)
- **Router Chain**: If X then A, else B

### 3. **LCEL** (LangChain Expression Language)
Modern way to compose chains using the `|` operator:

```python
chain = prompt | llm | output_parser
```

## Extension Points

### 1. Swap LLM Providers

**Easy**: Change one line to use different providers.

```python
# Local with Ollama
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.2")

# Cloud with Groq (fast)
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-70b-8192")

# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
```

**Why extend**: Switch between local/cloud, optimize for speed/cost/quality.

### 2. Add RAG (Retrieval-Augmented Generation)

**Purpose**: Let the LLM answer questions based on your documents.

**Components needed**:
- **Document Loader**: Load PDFs, websites, databases
- **Text Splitter**: Break documents into chunks
- **Embeddings**: Convert text to vectors
- **Vector Store**: Store and search embeddings
- **Retriever**: Find relevant documents

**Example use cases**:
- Chat with your PDFs
- Internal company knowledge base
- Customer support with product docs
- Research assistant

See: `example_rag.py`

### 3. Add Tools (Function Calling)

**Purpose**: Give the LLM capabilities beyond text generation.

**How it works**:
1. Define Python functions as tools
2. LLM decides when to use them
3. Execute functions and return results
4. LLM incorporates results into answer

**Example tools**:
- Calculator for math
- Web search for current info
- Database queries
- API calls (weather, stock prices, etc.)
- File operations
- Send emails/notifications

See: `example_tools.py`

### 4. Use Different Memory Types

**Purpose**: Control how conversation context is maintained.

| Memory Type | Best For | Pros | Cons |
|-------------|----------|------|------|
| **ConversationBufferMemory** | Short conversations | Full context | Memory grows unbounded |
| **ConversationBufferWindowMemory** | Long chats | Fixed size | Forgets old messages |
| **ConversationSummaryMemory** | Very long chats | Compressed | Loses details |
| **ConversationSummaryBufferMemory** | Production apps | Best of both | More complex |

See: `example_memory_output.py`

### 5. Structure Output with Parsers

**Purpose**: Get structured data instead of free text.

**Parser types**:
- **CommaSeparatedListOutputParser**: Get lists
- **StructuredOutputParser**: Get JSON with specific keys
- **PydanticOutputParser**: Get type-safe Python objects
- **Custom parsers**: Your own format

**Use cases**:
- Extract data from text
- Build APIs that return JSON
- Type-safe responses
- Data validation

See: `example_memory_output.py`

### 6. Compose Complex Chains

**Purpose**: Build multi-step workflows.

**Patterns**:

**Sequential** (A → B → C):
```python
chain = step1_chain | step2_chain | step3_chain
```
Use when: Each step needs previous step's output

**Parallel** (A, B, C at once):
```python
from langchain.schema.runnable import RunnableParallel

chain = RunnableParallel(
    task1=chain1,
    task2=chain2,
    task3=chain3
)
```
Use when: Independent analyses of same input

**Routing** (if X then A, else B):
```python
def route(input):
    if condition:
        return chain_a
    else:
        return chain_b
```
Use when: Different processing based on input type

See: `example_chains.py`

## Examples Included

| File | Demonstrates | Use Case |
|------|-------------|----------|
| `app.py` | Basic chat with memory | Interactive chatbot |
| `example_groq.py` | Cloud API usage | Fast cloud inference |
| `example_rag.py` | Document search | Chat with your docs |
| `example_tools.py` | Function calling | Calculator, search, etc. |
| `example_chains.py` | Chain composition | Multi-step workflows |
| `example_memory_output.py` | Memory & parsers | Structured responses |

## Common Patterns

### Pattern 1: Context-Aware Chatbot

```python
# Combines: Memory + Custom prompt + LLM
memory = ConversationBufferMemory()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant named {name}"),
    ("human", "{input}")
])
chain = prompt | llm | StrOutputParser()
```

**Extend to**:
- Add RAG for knowledge base
- Add tools for actions
- Add output parser for structured responses

### Pattern 2: Document Q&A System

```python
# Combines: Document loader + Embeddings + Vector store + Retriever + LLM
docs = load_documents("./docs")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
```

**Extend to**:
- Multiple document sources
- Reranking for better results
- Citation tracking
- Multi-query retrieval

### Pattern 3: Agent with Tools

```python
# Combines: Tools + Agent + Memory
tools = [calculator, search, database]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

**Extend to**:
- Custom tools for your domain
- Multi-agent collaboration
- Human-in-the-loop approval
- Tool error handling

### Pattern 4: Data Processing Pipeline

```python
# Combines: Sequential chains + Output parsers
extract_chain = prompt1 | llm | parser1
transform_chain = prompt2 | llm | parser2
load_chain = prompt3 | llm | parser3

pipeline = extract_chain | transform_chain | load_chain
```

**Extend to**:
- Parallel processing
- Error recovery
- Batch processing
- Progress tracking

## Best Practices

### 1. Start Simple, Add Complexity

```
✓ Start: Basic chat
→ Add: Memory
→ Add: Custom prompts
→ Add: RAG or tools
→ Add: Agents
```

### 2. Use LCEL for New Code

```python
# Old way (still works)
chain = LLMChain(llm=llm, prompt=prompt)

# New way (recommended)
chain = prompt | llm | output_parser
```

Benefits: Better streaming, parallel execution, easier debugging

### 3. Choose the Right Abstraction

| Need | Use |
|------|-----|
| Simple prompt + LLM | LCEL: `prompt \| llm` |
| Multiple steps | Sequential chain |
| LLM chooses tools | Agent |
| Search documents | RAG with retriever |
| Structured output | Output parser |

### 4. Test Components Individually

```python
# Test each piece
assert prompt.format(input="test") == expected
assert llm.invoke("test") != ""
assert parser.parse(output) == {"key": "value"}

# Then combine
chain = prompt | llm | parser
```

### 5. Handle Errors Gracefully

```python
try:
    result = chain.invoke(input)
except OutputParserException:
    # LLM didn't follow format
    result = retry_with_prompt
except Exception as e:
    # Other errors
    log_error(e)
    result = fallback_response
```

### 6. Monitor Performance

```python
# Add callbacks for logging
from langchain.callbacks import StdOutCallbackHandler

chain.invoke(input, config={"callbacks": [StdOutCallbackHandler()]})
```

### 7. Optimize for Your Model

**Local models (Ollama)**:
- Simpler prompts
- Fewer tools
- More explicit instructions
- Temperature = 0 for consistency

**Cloud models (GPT-4, Claude)**:
- Can handle complex prompts
- Multiple tools
- Function calling works well
- Higher temperature for creativity

## Real-World Extension Ideas

### 1. Personal Assistant
```
Base app
+ RAG (personal notes, calendar)
+ Tools (email, reminders, web search)
+ Memory (long-term context)
```

### 2. Customer Support Bot
```
Base app
+ RAG (product docs, FAQs)
+ Tools (ticket creation, order lookup)
+ Structured output (JSON responses for UI)
+ Routing (route to human if needed)
```

### 3. Code Assistant
```
Base app
+ RAG (codebase search)
+ Tools (run tests, format code, git)
+ Output parser (extract code blocks)
+ Chains (analyze → suggest → implement)
```

### 4. Research Assistant
```
Base app
+ RAG (research papers, notes)
+ Tools (web search, arxiv, wiki)
+ Chains (search → summarize → synthesize)
+ Memory (track research topics)
```

### 5. Content Generator
```
Base app
+ Sequential chains (outline → draft → edit)
+ Output parsers (extract metadata)
+ Tools (fact-checking, image gen)
+ Routing (different styles per content type)
```

## Next Steps

1. **Run the examples**: Try each example file to see patterns in action
2. **Modify examples**: Change prompts, models, parameters
3. **Combine patterns**: Mix RAG + tools, chains + memory, etc.
4. **Build your use case**: Start with the closest example and adapt

## Resources

- [LangChain Docs](https://python.langchain.com/)
- [LCEL Guide](https://python.langchain.com/docs/expression_language/)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [Example Apps](https://github.com/langchain-ai/langchain/tree/master/templates)

## Questions?

Common questions and where to find answers:

- **How do I...?** → Check the relevant example file
- **Which pattern should I use?** → See "Common Patterns" above
- **Local vs cloud?** → See "Swap LLM Providers"
- **Not working?** → See "Best Practices" #7 (Optimize for Your Model)
