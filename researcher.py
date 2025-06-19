import argparse
import os
from dataclasses import dataclass
from typing import List

import openai
from agents import Agent, Runner, function_tool
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma


@dataclass
class CodeIndexer:
    """Indexes a codebase into a Chroma vector store."""

    code_dir: str
    persist_dir: str = ".chroma_index"

    def __post_init__(self) -> None:
        self.embeddings = OpenAIEmbeddings()
        self.store = None

    def build(self) -> None:
        documents = []
        for root, _, files in os.walk(self.code_dir):
            for name in files:
                if not name.endswith((".py", ".md", ".txt")):
                    continue
                path = os.path.join(root, name)
                loader = TextLoader(path, encoding="utf-8")
                for doc in loader.load():
                    doc.metadata["source"] = os.path.relpath(path, self.code_dir)
                    documents.append(doc)
        if not documents:
            raise RuntimeError("No documents found to index")
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        docs = splitter.split_documents(documents)
        self.store = Chroma.from_documents(docs, self.embeddings, persist_directory=self.persist_dir)
        self.store.persist()

    def search(self, query: str, k: int = 4) -> str:
        if self.store is None:
            self.store = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        results = self.store.similarity_search(query, k=k)
        snippets: List[str] = []
        for doc in results:
            snippet = f"{doc.metadata.get('source', '')}:\n{doc.page_content}"
            snippets.append(snippet)
        return "\n---\n".join(snippets)


indexer: CodeIndexer | None = None


@function_tool
def search_code(query: str) -> str:
    """Search the indexed code for relevant snippets."""
    if indexer is None:
        raise RuntimeError("Indexer not initialized")
    return indexer.search(query)


@function_tool
def read_file(path: str) -> str:
    """Read a file from the codebase."""
    if indexer is None:
        raise RuntimeError("Indexer not initialized")
    # Prevent path traversal outside the indexed code directory
    full_path = os.path.normpath(os.path.join(indexer.code_dir, path))
    base_dir = os.path.realpath(indexer.code_dir)
    real_full_path = os.path.realpath(full_path)
    if not os.path.commonpath([real_full_path, base_dir]) == base_dir:
        raise ValueError("Invalid path")
    if not os.path.isfile(real_full_path):
        raise FileNotFoundError(path)
    with open(real_full_path, "r", encoding="utf-8") as f:
        return f.read()


def build_agent() -> Agent:
    instructions = (
        "You are Code Researcher, an autonomous agent that answers questions about"
        " large codebases. Use the search_code tool to retrieve relevant code\n"
        "snippets and read_file to inspect files in detail. Provide helpful,"
        " concise answers referencing file paths when possible."
    )
    return Agent(name="CodeResearcher", instructions=instructions, tools=[search_code, read_file])


def main() -> None:
    parser = argparse.ArgumentParser(description="Code Research Agent")
    parser.add_argument("--codebase", required=True, help="Path to the codebase")
    parser.add_argument("--question", required=True, help="Question to ask")
    args = parser.parse_args()

    global indexer
    indexer = CodeIndexer(args.codebase)
    indexer.build()

    agent = build_agent()
    result = Runner.run_sync(agent, args.question)
    print(result.final_output)


if __name__ == "__main__":
    main()
