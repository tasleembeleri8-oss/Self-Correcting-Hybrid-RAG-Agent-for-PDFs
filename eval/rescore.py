import json

from eval.run_eval import is_correct

with open("eval/questions.json", encoding="utf-8") as f:
    items = json.load(f)
with open("eval/results.json", encoding="utf-8") as f:
    details = json.load(f)

names = [k for k in details[0] if k != "question"]
n = len(items)
print("| Setup | Accuracy | Avg retries |")
print("|---|---|---|")
for name in names:
    correct = sum(is_correct(item, row[name]["answer"]) for item, row in zip(items, details))
    retries = sum(row[name]["retries"] for row in details)
    print(f"| {name} | {correct}/{n} ({100 * correct / n:.0f}%) | {retries / n:.2f} |")