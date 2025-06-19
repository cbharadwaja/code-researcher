import os
from dataclasses import dataclass
from typing import List

from langchain.document_loaders import TextLoader  # type: ignore[import-untyped]
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from .common import IndexNotReadyError


@dataclass
class CodeIndexer:
    """Indexes a codebase into a Chroma vector store."""

    code_dir: str
    persist_dir: str = ".chroma_index"
    allowed_extensions: List[str] | None = None

    def __post_init__(self) -> None:
        """Initializes OpenAI embeddings, prepares the store, and sets default file extensions."""
        self.embeddings = OpenAIEmbeddings()
        self.store = None  # Will be populated by build() or loaded by search()
        if self.allowed_extensions is None:
            self.allowed_extensions = [".py", ".md", ".txt"]

    def build(self) -> None:
        """
        Builds or rebuilds the vector index from files in `code_dir`.
        Persists the index to `persist_dir`, overwriting if it exists.
        Raises RuntimeError if no documents are found or no content is processable.
        """
        documents = []
        for root, _, files in os.walk(self.code_dir):
            for name in files:
                if not any(name.endswith(ext) for ext in self.allowed_extensions):
                    continue
                path = os.path.join(root, name)
                loader = TextLoader(path, encoding="utf-8")
                for doc in loader.load():
                    # Store relative path for cleaner references and smaller storage.
                    doc.metadata["source"] = os.path.relpath(path, self.code_dir)
                    documents.append(doc)
        if not documents:
            raise RuntimeError("No documents found to index in the specified codebase directory.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        docs = splitter.split_documents(documents)
        if not docs:
            raise RuntimeError("No processable content found in documents to build the index")
        self.store = Chroma.from_documents(docs, self.embeddings, persist_directory=self.persist_dir)
        self.store.persist()

    def search(self, query: str, k: int = 4) -> str:
        """
        Searches the index for a query. Loads from `persist_dir` if not already in memory.
        Raises IndexNotReadyError if the index cannot be loaded and wasn't built.
        """
        if self.store is None:
            try:
                if not os.path.exists(self.persist_dir) or not os.path.isdir(self.persist_dir):
                    raise FileNotFoundError(f"Persistence directory '{self.persist_dir}' not found. Please build the index first.")
                self.store = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
                # Here one might add a check if the loaded store is valid/not empty if Chroma API supports it.
                # For now, we rely on Chroma to raise an error if the directory is invalid or not a Chroma index.
            except Exception as e: # Catches FileNotFoundError and potential Chroma errors
                raise IndexNotReadyError(
                    f"Failed to load index from '{self.persist_dir}'. Ensure it's a valid index or build it first. Original error: {e}"
                ) from e

        if self.store is None: # Safeguard, should ideally be caught by logic above.
            raise IndexNotReadyError("Indexer store is not initialized. Please build the index.")

        results = self.store.similarity_search(query, k=k)
        snippets: List[str] = []
        for doc in results:
            snippet = f"{doc.metadata.get('source', '')}:\n{doc.page_content}"
            snippets.append(snippet)
        return "\n---\n".join(snippets)
