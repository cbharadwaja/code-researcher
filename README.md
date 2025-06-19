# Code Researcher Agent

This project implements a **Code Researcher Agent** inspired by the paper [Code Researcher: A Generalist Agent for Automated Code Research](https://arxiv.org/abs/2506.11060).

## Overview

The Code Researcher Agent is designed to autonomously analyze, understand, and answer questions about large codebases. It leverages advanced language models and retrieval-augmented generation (RAG) techniques to:
- Comprehend code structure and documentation
- Retrieve relevant code snippets and context
- Synthesize answers to user queries
- Support iterative research and code exploration

## Features
- **Automated Codebase Analysis**: Parses and indexes code for efficient retrieval.
- **Natural Language Querying**: Accepts user questions in plain English.
- **Contextual Code Retrieval**: Finds and presents relevant code sections.
- **Answer Synthesis**: Generates detailed, context-aware answers.
- **Iterative Research**: Supports follow-up questions and deeper exploration.

## Implementation Steps

1. **Set Up the Environment**
   - Install Python 3.9+ and pip.
   - (Optional) Set up a virtual environment:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate
     ```
   - Install required dependencies (see below).

2. **Install Dependencies**
   - Add the following to `requirements.txt`:
     - `openai` (or your preferred LLM API)
     - `langchain`
     - `tiktoken`
     - `faiss-cpu` or `chromadb` (for vector search)
     - `pygments` (for code parsing)
     - `rich` (for CLI output)
   - Install with:
     ```powershell
     pip install -r requirements.txt
     ```

3. **Codebase Indexing**
   - Recursively scan the target codebase.
   - Parse files, extract code, comments, and docstrings.
   - Chunk code and embed using an LLM embedding model.
   - Store embeddings in a vector database (e.g., FAISS, ChromaDB).

4. **Query Handling**
   - Accept user queries via CLI or API.
   - Embed the query and retrieve top-k relevant code chunks.
   - Use an LLM to synthesize an answer using the retrieved context.

5. **Iterative Research**
   - Maintain conversation history for follow-up questions.
   - Allow users to refine or expand queries.

## Example Usage

```powershell
python researcher.py --codebase ./my_project --question "How does authentication work?"
```

## References
- [Code Researcher: A Generalist Agent for Automated Code Research (arXiv:2506.11060)](https://arxiv.org/abs/2506.11060)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs/api-reference)

## License
This project is for research and educational purposes only. See the paper for more details on the Code Researcher agent design.
