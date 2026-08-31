
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path


def split_text(p):
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")   
    #same as meta-llama/Llama-3.1-8B model but without access restrictions , access can be requested throught huggingFace
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=250,
        chunk_overlap=50,
    )
    text = p.read_text(encoding="utf-8")
    text = json.loads(text)
    documents = []
    data = text["data"]
    parag = data[0]["paragraphs"][0]
    content = parag["context"]
        
        
    chunks = splitter.split_text(content)
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
        persist_directory="./chroma2__db"
    )

p = Path(r"C:\Internship\cuad\data\CUADv1_newer.json")
documents = split_text(p)
embedStore(documents)
