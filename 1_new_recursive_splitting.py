from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_PATH = Path(r"C:\Internship\cuad\data\CUADv1_newer.json")
OUTPUT_CHUNKS_PATH = Path(r"C:\Internship\temp\cuad_chunks_v2.json")
PERSIST_DIRECTORY = "./chroma1_new_db"


PAGE_MARKER_RE = re.compile(r"^\s*Page\s*-\d+-\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s*((?:\d+(?:\.\d+)*\.?|[A-Z])(?:\s+|\.\s+))(.+\S.*)$")


def normalize_contract_text(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.rstrip()
        if not line or PAGE_MARKER_RE.match(line):
            continue
        lines.append(re.sub(r"[ \t]+", " ", line))

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_title = "Preamble"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append((current_title, body))
        current_lines = []

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match and current_lines:
            flush()
            current_title = f"{match.group(1).strip()} {match.group(2).strip()}"
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush()
    return blocks


def split_block(title: str, body: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "; ", ". ", " ", ""],
        chunk_size=1800,
        chunk_overlap=250,
    )

    raw_chunks = splitter.split_text(body)
    documents: list[Document] = []

    for chunk_index, chunk in enumerate(raw_chunks, start=1):
        chunk_text = chunk.strip()
        if not chunk_text:
            continue

        if title and title != "Preamble":
            page_content = f"{title}\n\n{chunk_text}"
        else:
            page_content = chunk_text

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": "contract_1",
                    "original_title": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
                    "section_title": title,
                    "chunk_index": chunk_index,
                },
            )
        )

    return documents


def split_text(path: Path) -> list[Document]:
    text = json.loads(path.read_text(encoding="utf-8"))
    contract = text["data"][0]
    context = contract["paragraphs"][0]["context"]

    normalized = normalize_contract_text(context)
    blocks = split_into_blocks(normalized)

    documents: list[Document] = []
    for title, body in blocks:
        documents.extend(split_block(title, body))

    return documents


def save_chunks(documents: list[Document], output_path: Path) -> None:
    grouped = {
        "data": [
            {
                "source": "contract_1",
                "original_title": "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT",
                "splitted_text": [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in documents
                ],
            }
        ]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(grouped, ensure_ascii=False, indent=2), encoding="utf-8")


def embed_store(documents: list[Document]) -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
    )
    return vector_store


if __name__ == "__main__":
    documents = split_text(INPUT_PATH)
    save_chunks(documents, OUTPUT_CHUNKS_PATH)
    embed_store(documents)

    print(f"Saved {len(documents)} chunks to {OUTPUT_CHUNKS_PATH}")
    print(f"Embedded chunks into {PERSIST_DIRECTORY}")