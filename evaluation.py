from pathlib import Path
import json
import ollama

JUDGE_MODEL = "prometheus2"

PROMETHEUS_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing an evaluation criterion is given.
1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
[Does the response correctly and precisely extract the relevant clause(s) from the contract that answer the question, matching the reference answer in substance?]
Score 1: The response is completely unrelated to the reference answer, or extracts entirely wrong text.
Score 2: The response identifies the right general topic but misses the actual relevant clause, or extracts mostly irrelevant text.
Score 3: The response partially overlaps with the reference answer but is missing key parts or includes significant irrelevant text.
Score 4: The response closely matches the reference answer with only minor omissions or extra text.
Score 5: The response matches the reference answer exactly or is a fully correct paraphrase covering the same clause(s).

###Feedback:"""


def judge_answer(question: str, gold_answers: list[str], rag_answer: str) -> dict:
    reference = " | ".join(gold_answers) if gold_answers else "Not found in context."
    instruction = f"Highlight the parts of this contract that answer: {question}"

    prompt = PROMETHEUS_PROMPT.format(
        instruction=instruction,
        response=rag_answer,
        reference=reference,
    )

    response = ollama.generate(model=JUDGE_MODEL, prompt=prompt, options={"temperature": 0.1})
    raw_text = response["response"].strip()

    score = None
    feedback = raw_text
    if "[RESULT]" in raw_text:
        feedback, _, score_part = raw_text.partition("[RESULT]")
        feedback = feedback.strip()
        digits = "".join(c for c in score_part if c.isdigit())
        if digits:
            score = int(digits[0])

    return {"score": score, "feedback": feedback, "raw": raw_text}


# ---- Run judge over rag_results.jsonl ----
in_path = Path(r"C:\Internship\cuad\data\rag_results1_v2.jsonl")
out_path = Path(r"C:\Internship\cuad\data\judged_results_1_v2.jsonl")

with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        record = json.loads(line)
        judgment = judge_answer(
            record["question"],
            record["gold_answers"],
            record["rag_model_response"],
        )
        record["judge_score"] = judgment["score"]
        record["judge_feedback"] = judgment["feedback"]
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{record['question']} -> score {judgment['score']}")