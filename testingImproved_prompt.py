from pathlib import Path
import json
import re
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

# ---- question parsing ----
CATEGORY_RE = re.compile(r'related to "([^"]+)"')
DETAILS_RE = re.compile(r'Details:\s*(.+)$')

def parse_question(question: str):
    cat_match = CATEGORY_RE.search(question)
    det_match = DETAILS_RE.search(question)
    category = cat_match.group(1).strip() if cat_match else question.strip()
    details = det_match.group(1).strip() if det_match else question.strip()
    if not det_match:
        print(f"[parse warning] no 'Details:' found in: {question!r}")
    return category, details

# ---- RAG setup ----
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    collection_name="langchain",
    embedding_function=embedding_model,
    persist_directory=r"C:\Internship\chroma3__db",
)

OLLAMA_MODEL = "llama3.1:8b"
TOP_K = 5

def rag_query(question: str, top_k: int = TOP_K):
    category, details = parse_question(question)

    # keep embedding the full question for retrieval — richer signal than either half alone
    docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""You are extracting exact clauses from a legal contract for lawyer review.

Each question has two parts:
- Category: a general label for context only.
- What to extract: the specific instruction describing exactly what
  information must be found. This is the operative instruction — always
  extract based on "What to extract", not just the Category name.

Rules:
- Copy the relevant text EXACTLY as it appears in the context below. Do not paraphrase, summarize, or explain.
- Only extract the specific clause(s) or fact(s) that satisfy "What to extract" — not surrounding sentences that aren't relevant.
- If "What to extract" asks for a specific fact (a date, name, amount, duration, etc.), extract the shortest exact span containing that fact, not the whole surrounding clause.
- If there are multiple relevant clauses, separate them with " | ".
- If the context does not contain text satisfying "What to extract", respond with exactly: Not found in context.
- Do not add any commentary, labels, or explanation before or after the extracted text.

Example:
Category: Widget Delivery Deadline
What to extract: The number of days the supplier has to deliver widgets after receiving an order
Context: "Supplier shall deliver all ordered Widgets within thirty (30) business days of receipt of a Purchase Order, time being of the essence."
Extracted clause(s): within thirty (30) business days of receipt of a Purchase Order

Context:
{context}

Category: {category}
What to extract: {details}
Extracted clause(s):"""

    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    return category, details, response["response"].strip()

# ---- Run RAG over each question, write incrementally ----
out_path = Path(r"C:\Internship\cuad\data\rag_results_test_improv_for_chroma3.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for question, gold_answers in question_ans:
        category, details, rag_answer = rag_query(question)
        row = {
            "question": question,
            "category": category,
            "details": details,
            "gold_answers": gold_answers,
            "rag_model_response": rag_answer
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(question, "->", rag_answer)