import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
#pip install langchain-experimental


def split_text(p):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    semantic_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )

    text = p.read_text(encoding="utf-8")
    text = json.loads(text)
    documents = []
    data = text["data"]
    parag = data[0]["paragraphs"][0]
    content = parag["context"]
    
    
    chunks = semantic_splitter.split_text(content)
    for chunk in chunks:
        documents.append(Document(page_content=chunk, metadata={"contract_id": "contract_1"}))

    return documents
    
def embedStore(documents):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs = {"device": "cuda"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    vecor_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="./chroma3__db"
    )

p = Path(r"C:\Internship\cuad\data\CUADv1_newer.json")
documents = split_text(p)
embedStore(documents)

