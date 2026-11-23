"""Run synthetic five-factor ranking using only the Python standard library."""

import json
from pathlib import Path


def main():
    sample_path = Path(__file__).with_name("synthetic_candidates.json")
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    weights = sample["weights"]
    ranked = sorted(
        sample["candidates"],
        key=lambda candidate: sum(
            weights[key] * candidate["scores"][key] for key in weights
        ),
        reverse=True,
    )
    print("Synthetic demonstration only; scores are hand-authored, not model predictions.")
    for rank, candidate in enumerate(ranked[:3], start=1):
        score = sum(weights[key] * candidate["scores"][key] for key in weights)
        print(f"{rank}. {candidate['title']} | score={score:.2f}")


if __name__ == "__main__":
    main()
