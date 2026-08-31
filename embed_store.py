"""
Step 3 & 4 of the RAG pipeline for CUAD:
  - Load the previously split chunks from the saved JSON file
    (produced by load_and_split_cuad.py -> save_chunks()).
  - Embed every chunk locally on GPU using nomic-ai/nomic-embed-text-v1.5,
    and store them in a single, persisted Chroma vector database,
    tagged with per-contract metadata for later filtering.

Why nomic-embed-text-v1.5 instead of jina-embeddings-v3:
  Comparable quality and long-context support (8192 tokens), but licensed
  Apache 2.0 (fully permissive) instead of CC BY-NC 4.0 (non-commercial only)
  -- important since this project may end up in a public portfolio/LinkedIn.

Requirements:
    pip install sentence-transformers torch einops langchain-chroma chromadb

Usage:
    python embed_and_store_cuad.py path/to/cuad_chunks.json
"""

import json
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer


PERSIST_DIRECTORY = "./cuad_chroma_db"


class NomicEmbeddings(Embeddings):
    """
    Wraps nomic-ai/nomic-embed-text-v1.5 for use as a LangChain-compatible
    embedding function, running locally on GPU.

    Nomic requires a task instruction PREFIX on every input string (not a
    separate parameter like Jina) -- documents get "search_document: " and
    queries get "search_query: ". Using the wrong prefix (or none) noticeably
    hurts retrieval quality, so embed_documents() and embed_query() apply
    different prefixes deliberately.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "cuda"):
        # trust_remote_code=True is required on current sentence-transformers/transformers
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"search_query: {text}"
        embedding = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].tolist()


def load_chunks(input_path: str) -> list[Document]:
    """
    Reload previously split chunks from the grouped JSON format
    (data -> [ {source, original_title, splitted_text: [...]} ])
    back into a flat list of Document objects.
    """
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    documents = []
    for entry in raw["data"]:
        for chunk in entry["splitted_text"]:
            documents.append(
                Document(page_content=chunk["page_content"], metadata=chunk["metadata"])
            )

    return documents


def embed_and_store(chunks: list[Document], persist_directory: str = PERSIST_DIRECTORY) -> Chroma:
    """
    Embeds all chunks locally via Nomic and stores them in a single
    Chroma collection, persisted to disk. If a DB already exists at
    persist_directory, it is loaded instead of re-embedding everything
    from scratch.
    """
    embeddings = NomicEmbeddings(device="cuda")

    if Path(persist_directory).exists():
        print(f"Found existing Chroma DB at '{persist_directory}' -- loading it instead of re-embedding.")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
    else:
        print(f"No existing DB found. Embedding {len(chunks)} chunks locally on GPU -- this may take a while.")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        print(f"Done. Persisted to '{persist_directory}'.")

    return vectorstore


if __name__ == "__main__":
    chunks_json_path = r"C:\Internship\temp\cuad_chunks.json"

    if not Path(chunks_json_path).exists():
        print(f"Could not find {chunks_json_path}")
        print("Pass the correct path as an argument, e.g.:")
        print("  python embed_and_store_cuad.py cuad_chunks.json")
        sys.exit(1)

    chunks = load_chunks(chunks_json_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_json_path}")

    vectorstore = embed_and_store(chunks)

    # Quick sanity check: run one similarity search to confirm it works end-to-end
    results = vectorstore.similarity_search("governing law", k=2)
    print(f"\nSanity check -- top result for 'governing law':")
    print(f"  source: {results[0].metadata.get('source')}")
    print(f"  content preview: {results[0].page_content[:150]}...")