# LangChain Practice

A practice repository implementing examples from the official LangChain documentation to learn about building AI agents, chat models, and semantic search engines.

## Overview

This project demonstrates key LangChain concepts through three example scripts:
- `agent.py`: A ReAct (Reasoning + Acting) agent with memory that can perform web searches using Tavily and process user queries
- `chat_model.py`: Chat model implementation showing prompt templates, streaming, and different input formats
- `semantic_search_engine.py`: Document processing pipeline with PDF loading, text splitting, embedding generation, and vector search

## Dependencies

```
langchain_community==0.3.20
langchain_core==0.3.49
langchain_mistralai==0.2.10
langchain_postgres==0.0.13
langchain_text_splitters==0.3.7
langgraph==0.3.21
python-dotenv==1.1.0
```

Additional dependencies: `pypdf` for PDF loading, PostgreSQL with pgvector extension (for vector storage)

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file with required API keys:
   ```
   MISTRAL_API_KEY=your_key
   LANGSMITH_API_KEY=your_key  # Optional for tracing
   LANGSMITH_PROJECT=default   # Optional project name
   TAVILY_API_KEY=your_key     # For web search capabilities
   ```

## Usage

Each script demonstrates different LangChain capabilities:

### Agent
```python
# Create a ReAct agent with memory
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Run with conversation persistence
config = {"configurable": {"thread_id": "conversation_id"}}
response = agent_executor.invoke({"messages": [HumanMessage(content="What is the weather in Delhi?")]}, config)
```

### Chat Model
```python
# Using prompt templates
prompt = prompt_template.invoke({"language": "French", "text": "Hello world"})
response = model.invoke(prompt)
```

### Semantic Search
```python
# Search documents
results = vector_store.similarity_search("How many distribution centers does Nike have?")
```

## Resources

For detailed explanations of these concepts, refer to the [official LangChain documentation](https://python.langchain.com/docs/get_started/introduction).

**Note:** Don't try to mess with the postgres connection URL - the DB is already removed so don't act smart.