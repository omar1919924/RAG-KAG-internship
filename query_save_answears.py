from pathlib import Path
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

# ---- existing parsing code ----
p = Path(r"C:\Internship\cuad\data\CUADv1_newer.json")
text = json.loads(p.read_text(encoding="utf-8"))

question_ans = []
data = text["data"]
parag = data[0]["paragraphs"][0]["qas"]

for entry in parag:
    if entry["is_impossible"] == False:
        cleaned = [' '.join(a['text'].split()) for a in entry['answers']]
        cleaned = list(dict.fromkeys(cleaned))  # dedupe
        question_ans.append((entry["question"], cleaned))

# ---- RAG setup ----
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    collection_name="langchain",
    embedding_function=embedding_model,
    persist_directory=r"C:\Internship\chroma2__db",
)

OLLAMA_MODEL = "llama3.1:8b"
TOP_K = 5

def rag_query(question: str, top_k: int = TOP_K) -> str:
    docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""You are extracting exact clauses from a legal contract for lawyer review.

Rules:
- Copy the relevant text EXACTLY as it appears in the context below. Do not paraphrase, summarize, or explain.
- Only extract the specific clause(s) that answer the question — not surrounding sentences that aren't relevant.
- If there are multiple relevant clauses, separate them with " | ".
- If no part of the context answers the question, respond with exactly: Not found in context.
- Do not add any commentary, labels, or explanation before or after the extracted text.

Context:
{context}

Question: {question}
Extracted clause(s):"""

    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response["response"].strip()

# ---- Run RAG over each question, write incrementally ----
out_path = Path(r"C:\Internship\cuad\data\rag_results2.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for question, gold_answers in question_ans:
        rag_answer = rag_query(question)
        row = {
            "question": question,
            "gold_answers": gold_answers,
            "rag_model_response": rag_answer
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(question, "->", rag_answer)