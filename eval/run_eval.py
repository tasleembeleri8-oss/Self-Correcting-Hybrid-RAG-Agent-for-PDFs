"""Compare: vector only vs hybrid vs hybrid + self-correction, on ONE document.

Usage (from project root):
    python -m eval.run_eval                      # uses the first PDF in data/pdfs/
    python -m eval.run_eval path/to/file.pdf     # or a specific PDF
questions.json must contain questions about that PDF.
"""
import json
import re
import sys
import time
from pathlib import Path

from app.documents import add_pdf
from app.graph import agent, llm, NOT_FOUND
from app.retriever import hybrid_search, vector_search

IDK_PHRASES = ["don't know", "do not know", "not in the context", "does not contain",
               "doesn't contain", "no information", "not mentioned", "cannot answer",
               "not available"]


def norm(text: str) -> str:
    t = text.lower().replace("\u2019", "'").replace("\u2011", "-").replace("*", "")
    return re.sub(r"[\s\u00a0\u202f\u2009]+", " ", t).strip()


def answer_simple(doc_id, question, search_fn):
    """Baselines: one search, then generate. No grading, no rewriting."""
    ctx = "\n---\n".join(search_fn(doc_id, question))
    ans = llm.invoke(
        "Answer the question using ONLY the context below. "
        f"If the answer is not in the context, reply exactly: {NOT_FOUND}\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}"
    ).content
    return ans, 0


def answer_agent(doc_id, question):
    r = agent.invoke({"doc_id": doc_id, "question": question, "query": question, "retries": 0})
    return r["answer"], r["retries"]


def safe_call(fn, *args, tries=4):
    for t in range(tries):
        try:
            return fn(*args)
        except Exception as e:
            print("   error, waiting and retrying:", str(e)[:100])
            time.sleep(15 * (t + 1))
    return "ERROR", 0


def is_correct(item, answer):
    a = norm(answer)
    if not item["answerable"]:
        return any(p in a for p in IDK_PHRASES)
    if "expected_all" in item:
        return all(norm(x) in a for x in item["expected_all"])
    return any(norm(x) in a for x in item["expected_any"])


CONFIGS = {
    "Vector only": lambda d, q: safe_call(answer_simple, d, q, vector_search),
    "Hybrid (vector + BM25)": lambda d, q: safe_call(answer_simple, d, q, hybrid_search),
    "Hybrid + self-correction": lambda d, q: safe_call(answer_agent, d, q),
}


def main():
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("data/pdfs").glob("*.pdf"))[0]
    doc = add_pdf(pdf.read_bytes(), pdf.name)      # the same pipeline the /upload endpoint uses
    print(f"Indexed {pdf.name}: {doc.pages} pages, {len(doc.chunks)} chunks\n")

    with open("eval/questions.json", encoding="utf-8") as f:
        items = json.load(f)

    stats = {name: {"correct": 0, "retries": 0} for name in CONFIGS}
    details = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['question']}")
        row = {"question": item["question"]}
        for name, run in CONFIGS.items():
            ans, retries = run(doc.id, item["question"])
            ok = is_correct(item, ans)
            stats[name]["correct"] += ok
            stats[name]["retries"] += retries
            row[name] = {"correct": ok, "retries": retries, "answer": ans}
            print(f"   {name}: {'PASS' if ok else 'FAIL'}")
            time.sleep(2)
        details.append(row)

    n = len(items)
    print("\n| Setup | Accuracy | Avg retries |")
    print("|---|---|---|")
    for name, s in stats.items():
        print(f"| {name} | {s['correct']}/{n} ({100 * s['correct'] / n:.0f}%) | {s['retries'] / n:.2f} |")

    with open("eval/results.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print("\nFull answers saved to eval/results.json")


if __name__ == "__main__":
    main()
