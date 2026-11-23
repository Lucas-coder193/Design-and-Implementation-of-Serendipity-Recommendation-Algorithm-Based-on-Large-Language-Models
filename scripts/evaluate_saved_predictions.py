"""Recompute ranking metrics from local saved movie predictions.

Full datasets and predictions are deliberately not distributed with this repo.
Run from any directory after restoring the files to their original paths.
"""

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    ("random", "随机采样合适参数/topk_final_test_set.jsonl",
     "随机采样系统推荐测试/test_analyst_generated_predictions.jsonl",
     "随机采样系统推荐测试/test_explorer_generated_predictions.jsonl",
     [2.0, 3.5, 1.0, 2.5, 1.0]),
    ("target_similar", "细粒度合适参数/topk_hard_final_test_set.jsonl",
     "细粒度系统推荐测试/test_hard_analyst_generated_predictions.jsonl",
     "细粒度系统推荐测试/test_hard_explorer_generated_predictions.jsonl",
     [2.0, 3.0, 1.0, 2.5, 1.5]),
    ("history_similar", "信息茧房合适参数/topk_last_final_test_set.jsonl",
     "信息茧房系统推荐测试/test_last_analyst_generated_predictions.jsonl",
     "信息茧房系统推荐测试/test_last_explorer_generated_predictions.jsonl",
     [2.5, 1.0, 1.5, 1.5, 3.5]),
]


def read_jsonl(path):
    with (ROOT / path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def parse_prediction(row):
    value = row["predict"]
    return json.loads(value) if isinstance(value, str) else value


def evaluate(groups, analyst, explorer, weights):
    count = sum(len(group["candidates"]) for group in groups)
    if not groups or len(analyst) != count or len(explorer) != count:
        raise ValueError("Candidate groups and predictions must have equal, nonzero lengths.")
    totals = dict.fromkeys(["HR@3", "HR@5", "HR@10", "NDCG@10", "MRR", "AUC"], 0.0)
    pointer = 0
    for group in groups:
        scored = []
        for candidate in group["candidates"]:
            marker = f"Target Item: 《{candidate['title']}》"
            if any(marker not in row["prompt"] for row in (analyst[pointer], explorer[pointer])):
                raise ValueError(f"Prediction alignment mismatch at row {pointer}.")
            a = parse_prediction(analyst[pointer])
            e = parse_prediction(explorer[pointer])
            vector = [a["relevance"], a["utility"], e["novelty"], e["unexpectedness"], e["diversity"]]
            score = sum(float(value) * weight for value, weight in zip(vector, weights))
            scored.append((score, candidate["is_label"]))
            pointer += 1
        ranked = sorted(scored, key=lambda row: row[0], reverse=True)
        ranks = [i + 1 for i, (_, label) in enumerate(ranked) if label == 1]
        positives = [score for score, label in scored if label == 1]
        negatives = [score for score, label in scored if label == 0]
        if not positives or not negatives:
            raise ValueError("Each group must contain both positive and negative candidates.")
        for k in (3, 5, 10):
            totals[f"HR@{k}"] += int(any(rank <= k for rank in ranks))
        dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(10, len(ranks))))
        totals["NDCG@10"] += dcg / idcg
        totals["MRR"] += 1 / min(ranks)
        totals["AUC"] += sum(
            (p > n) + 0.5 * (p == n) for p in positives for n in negatives
        ) / (len(positives) * len(negatives))
    return {
        "groups": len(groups),
        "distinct_users": len({group["user_id"] for group in groups}),
        "items": count,
        "weights": weights,
        "metrics": {key: round(value / len(groups), 6) for key, value in totals.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional output JSON file")
    args = parser.parse_args()
    report = {
        "source": "Recomputed from locally saved predictions; no model training or inference rerun.",
        "weight_order": ["relevance", "utility", "novelty", "unexpectedness", "diversity"],
        "experiments": {},
    }
    for name, dataset, analyst_path, explorer_path, weights in EXPERIMENTS:
        groups, analyst, explorer = map(read_jsonl, (dataset, analyst_path, explorer_path))
        report["experiments"][name] = evaluate(groups, analyst, explorer, weights)
        if name == "history_similar":
            for ablation, ablation_weights in [
                ("history_similar_analyst_only", [1.0, 9.0, 0.0, 0.0, 0.0]),
                ("history_similar_explorer_only", [0.0, 0.0, 1.0, 1.0, 8.0]),
            ]:
                report["experiments"][ablation] = evaluate(groups, analyst, explorer, ablation_weights)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
