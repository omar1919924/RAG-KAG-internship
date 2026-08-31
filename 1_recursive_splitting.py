from langchain_text_splitters  import RecursiveCharacterTextSplitter
from pathlib import Path
import json
from langchain_chroma  import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sqlalchemy import text


#load and split the text into chunks
def split_text(p):
    splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", " ", ""],
        chunk_size = 1000,
        chunk_overlap = 200
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
    

#vector embedding and store in chroma db
def embedStore(documents):
    embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)
    vector_store = Chroma.from_documents(
    documents=documents,
    embedding= embedding_model,
    persist_directory= "./chroma1_db")




    
    



p = Path(r"C:\Internship\cuad\data\CUADv1_newer.json")
documents = split_text(p)
embedStore(documents)
