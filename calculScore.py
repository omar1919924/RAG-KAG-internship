from pathlib import Path
import json
import statistics
from collections import Counter

in_path = Path(r"C:\Internship\cuad\data\judged_results1_v4.jsonl")

records = []
with in_path.open("r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# Garder seulement les scores valides (exclure les None / parsing échoué)
scores = [r["judge_score"] for r in records if r.get("judge_score") is not None]
missing = len(records) - len(scores)

print(f"Total questions: {len(records)}")
print(f"Scores valides: {len(scores)}")
print(f"Scores manquants (parsing échoué): {missing}")
print()

if scores:
    print(f"Moyenne: {statistics.mean(scores):.2f}")
    print(f"Médiane: {statistics.median(scores)}")
    print(f"Écart-type: {statistics.stdev(scores):.2f}" if len(scores) > 1 else "")
    print(f"Min / Max: {min(scores)} / {max(scores)}")
    print()

    # Distribution par score (1 à 5)
    dist = Counter(scores)
    print("Distribution des scores:")
    for s in range(1, 6):
        count = dist.get(s, 0)
        pct = 100 * count / len(scores)
        bar = "█" * count
        print(f"  {s}: {count:3d} ({pct:5.1f}%) {bar}")
else:
    print("Aucun score valide trouvé.")