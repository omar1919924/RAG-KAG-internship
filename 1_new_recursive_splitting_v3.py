from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_PATH = Path(r"C:\Internship\cuad\data\CUADv1_newer.json")
OUTPUT_CHUNKS_PATH = Path(r"C:\Internship\temp\cuad_chunks_v3.json")
PERSIST_DIRECTORY = "./chroma1_new_db_v3"

PAGE_MARKER_RE = re.compile(r"^\s*Page\s*-\d+-\s*$", re.IGNORECASE)
SECTION_LINE_RE = re.compile(r"^\s*(?P<prefix>(?:\d+(?:\.\d+)*|[A-Z]))(?:\.|\s+)\s*(?P<rest>.+)$")


def normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    previous_line = None

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line.strip())
        if not line or PAGE_MARKER_RE.match(line):
            continue
        if line == previous_line:
            continue
        lines.append(line)
        previous_line = line

    return lines


def split_heading_line(line: str) -> tuple[str | None, str | None]:
    match = SECTION_LINE_RE.match(line)
    if not match:
        return None, None

    prefix = match.group("prefix")
    rest = match.group("rest").strip()
    if not rest:
        return f"{prefix}", None

    if ". " in rest:
        title_text, body = rest.split(". ", 1)
        return f"{prefix} {title_text}.", body.strip()

    return f"{prefix} {rest}", None


def build_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    current_title = "Preamble"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body:
            blocks.append((current_title, body))
        current_lines = []

    for line in normalize_lines(text):
        heading, body = split_heading_line(line)
        if heading is not None:
            flush()
            current_title = heading
            if body:
                current_lines.append(body)
            continue

        current_lines.append(line)

    flush()
    return blocks


def split_block(title: str, body: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        chunk_size=900,
        chunk_overlap=150,
    )

    chunks = splitter.split_text(body)
    documents: list[Document] = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        cleaned_chunk = chunk.strip()
        if not cleaned_chunk:
            continue

        page_content = f"{title}\n\n{cleaned_chunk}" if title != "Preamble" else cleaned_chunk
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
    context = text["data"][0]["paragraphs"][0]["context"]

    documents: list[Document] = []
    for title, body in build_blocks(context):
        documents.extend(split_block(title, body))

    return documents


def save_chunks(documents: list[Document], output_path: Path) -> None:
    payload = {
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
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def embed_store(documents: list[Document]) -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
    )


if __name__ == "__main__":
    documents = split_text(INPUT_PATH)
    save_chunks(documents, OUTPUT_CHUNKS_PATH)
    embed_store(documents)

    print(f"Saved {len(documents)} chunks to {OUTPUT_CHUNKS_PATH}")
    print(f"Embedded chunks into {PERSIST_DIRECTORY}")