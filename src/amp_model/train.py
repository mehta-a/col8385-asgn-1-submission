"""Trivial reference "training": an empirical amino-acid-frequency and length-histogram table.

Stands in for a real generative model — the point is exercising the grader's train/generate
contract (checkpoint/ produced fresh by `uv run train`, never committed to the repo), not AMP
quality.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _read_fasta_sequences(path: Path) -> list[str]:
    sequences: list[str] = []
    parts: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if parts:
                sequences.append("".join(parts))
                parts = []
        else:
            parts.append(line.upper())
    if parts:
        sequences.append("".join(parts))
    return sequences


def train(training_fasta: Path) -> dict:
    sequences = _read_fasta_sequences(training_fasta)
    if not sequences:
        raise ValueError(f"No sequences found in {training_fasta}")

    aa_counts: Counter[str] = Counter()
    length_counts: Counter[int] = Counter()
    for seq in sequences:
        aa_counts.update(seq)
        length_counts[len(seq)] += 1

    total_aa = sum(aa_counts.values())
    total_len = sum(length_counts.values())
    return {
        "aa_freqs": {aa: c / total_aa for aa, c in sorted(aa_counts.items())},
        "length_probs": {str(length): c / total_len for length, c in sorted(length_counts.items())},
        "n_train_sequences": len(sequences),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-fasta",
        type=Path,
        default=Path("data/training/training.fasta"),
        help="Overwritten by the grader at grade time; point this at your own local copy to self-test.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoint"))
    args = parser.parse_args()

    if not args.training_fasta.exists():
        print(f"ERROR: training data not found at {args.training_fasta}", file=sys.stderr)
        sys.exit(1)

    checkpoint = train(args.training_fasta)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "model.json"
    out_path.write_text(json.dumps(checkpoint, indent=2))
    print(f"Trained on {checkpoint['n_train_sequences']} sequences -> {out_path}")


if __name__ == "__main__":
    main()