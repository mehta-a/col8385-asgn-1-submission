#!/usr/bin/env python3
"""
bench.py - evaluate several model.pt checkpoints and rank them on one leaderboard.

For every (checkpoint, seed) pair this generates a library, scores it with the
grader's scorer.py, and then ranks everything using the grader's own
aggregate_scores.py, imported directly so the arithmetic cannot drift from the
instructor's.

Each entry carries the checkpoint path, its md5 and the epoch it came from, so
a leaderboard row can always be traced back to the exact weights that produced
it. That is the fix for checkpoints getting mixed up.

Running more than one seed per checkpoint is the point rather than a nicety:
the report includes a variance section comparing the spread between seeds of
the same model against the spread between models. If the first is comparable
to the second, the ranking is noise and the report says so.

Usage
-----
    python bench.py --models runs/v2/snapshot-e*.pt --seeds 42 43 \\
        --grader ../amp-grader-demo --out bench/leaderboard.json

    # reuse metrics already on disk, just redo the ranking
    python bench.py --models runs/v2/snapshot-e*.pt --seeds 42 43 \\
        --grader ../amp-grader-demo --out bench/leaderboard.json --skip-generate
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Import the grader's aggregation code so the maths is theirs, not ours
# ---------------------------------------------------------------------------

def load_aggregator(grader: Path):
    path = grader / "scripts" / "aggregate_scores.py"
    if not path.exists():
        raise SystemExit(f"cannot find {path}. Pass --grader pointing at amp-grader-demo.")
    spec = importlib.util.spec_from_file_location("aggregate_scores", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def epoch_of(path: Path) -> int | None:
    match = re.search(r"e(\d+)", path.stem)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Generate and score one library
# ---------------------------------------------------------------------------

def run(command, cwd=None):
    print("  $ " + " ".join(str(c) for c in command))
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"command failed with exit code {result.returncode}")


def generate_library(model: Path, library: Path, seed: int, n: int, submission: Path):
    run(["uv", "run", "generate",
         "--model", str(model.resolve()),
         "--out", str(library.resolve()),
         "--seed", str(seed),
         "--n-sequences", str(n)],
        cwd=submission)


def score_library(library: Path, metrics: Path, grader: Path, profile: str):
    run(["uv", "run", "--project", ".", "python", "scorer/scorer.py",
         "--library-fasta", str(library.resolve()),
         "--profile", profile,
         "--training-fasta", "data/training/training.fasta",
         "--reference-amps-fasta", "data/training/training.fasta",
         "--reference-generic-fasta", "data/generic/background.fasta",
         "--out", str(metrics.resolve())],
        cwd=grader)


# ---------------------------------------------------------------------------
# Variance: is the ranking measuring the model or the sampler?
# ---------------------------------------------------------------------------

def variance_report(entries: dict) -> dict:
    by_model: dict[str, dict[str, list[float]]] = {}
    for entry in entries.values():
        by_model.setdefault(entry["model"], {})
        for metric, value in entry["raw"].items():
            if isinstance(value, (int, float)):
                by_model[entry["model"]].setdefault(metric, []).append(value)

    metrics = sorted({m for values in by_model.values() for m in values})
    out = {}
    for metric in metrics:
        within = []
        model_means = []
        for values in by_model.values():
            series = values.get(metric, [])
            if len(series) >= 2:
                within.append(max(series) - min(series))
            if series:
                model_means.append(statistics.fmean(series))
        if not model_means:
            continue
        across = max(model_means) - min(model_means)
        worst_within = max(within) if within else None
        verdict = "unknown (need at least 2 seeds per model)"
        if worst_within is not None:
            if across == 0:
                verdict = "no difference between models"
            elif worst_within >= across:
                verdict = "NOISE: seed spread exceeds model spread"
            elif worst_within >= across * 0.5:
                verdict = "marginal: seed spread is over half the model spread"
            else:
                verdict = "real: models differ by more than seed noise"
        out[metric] = {
            "spread_between_seeds_worst_case": worst_within,
            "spread_between_models": across,
            "verdict": verdict,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rank model checkpoints on one leaderboard.")
    parser.add_argument("--models", nargs="+", required=True, help="model.pt checkpoints")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--grader", type=Path, default=Path("../amp-grader-demo"))
    parser.add_argument("--submission", type=Path, default=Path("."),
                        help="submission repo root, where `uv run generate` works")
    parser.add_argument("--work", type=Path, default=Path("bench"),
                        help="where libraries and metrics files are written")
    parser.add_argument("--out", type=Path, default=Path("bench/leaderboard.json"))
    parser.add_argument("--n-sequences", type=int, default=1000)
    parser.add_argument("--profile", default="grader")
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--skip-generate", action="store_true",
                        help="reuse libraries and metrics already in --work")
    args = parser.parse_args()

    aggregator = load_aggregator(args.grader)
    args.work.mkdir(parents=True, exist_ok=True)

    entries = {}
    for model_path in args.models:
        model = Path(model_path)
        if not model.exists():
            raise SystemExit(f"checkpoint not found: {model}")
        digest = file_md5(model)
        for seed in args.seeds:
            tag = f"{model.stem}_seed{seed}"
            library = args.work / f"{tag}.fasta"
            metrics = args.work / f"{tag}.metrics.json"

            print(f"\n=== {tag}  ({model}, md5 {digest[:8]}) ===")
            if not args.skip_generate:
                generate_library(model, library, seed, args.n_sequences, args.submission)
                score_library(library, metrics, args.grader, args.profile)
            if not metrics.exists():
                print(f"  no metrics at {metrics}, skipping")
                continue

            entries[tag] = {
                "model": model.stem,
                "checkpoint": str(model.resolve()),
                "checkpoint_md5": digest,
                "epochs": epoch_of(model),
                "seed": seed,
                "library": str(library),
                "metrics_file": str(metrics),
                "report": json.loads(metrics.read_text()),
            }

    if len(entries) < 1:
        raise SystemExit("nothing scored")

    # Rank with the grader's own code, on exactly the reports we just built.
    reports = {tag: entry["report"] for tag, entry in entries.items()}
    objectives = dict(aggregator.check_consistent_objectives(reports))
    raw = aggregator.extract_raw_values(reports)
    aggregator.add_fbd_margin(raw, objectives)
    components = [m for m in objectives if m not in aggregator.CONSTRAINT_METRICS]
    fixed = aggregator.direction_fix(raw, objectives)
    normalized = aggregator.normalize_components(fixed, components, args.epsilon)

    for tag, entry in entries.items():
        entry["raw"] = raw[tag]
        entry["components"] = normalized[tag]
        entry["aggregate_score"] = aggregator.geometric_mean(
            [normalized[tag][m] for m in components]
        )
        entry.pop("report")

    ranking = sorted(entries.items(), key=lambda kv: -kv[1]["aggregate_score"])
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "grader": str(args.grader.resolve()),
        "profile": args.profile,
        "seeds": args.seeds,
        "n_sequences": args.n_sequences,
        "components": components,
        "objectives": objectives,
        "entries": entries,
        "ranking": [[tag, entry["aggregate_score"]] for tag, entry in ranking],
        "variance": variance_report(entries),
        "note": (
            "aggregate_score is normalised against this set of entries only. With "
            "checkpoints of one model it amplifies small differences in the unbounded "
            "metrics. Read the variance section before trusting the ranking."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print("\n=== leaderboard " + "=" * 46)
    print("%-26s %10s  %s" % ("entry", "aggregate", "md5"))
    print("-" * 60)
    for tag, entry in ranking:
        print("%-26s %10.4f  %s" % (tag, entry["aggregate_score"], entry["checkpoint_md5"][:8]))

    noisy = [m for m, v in payload["variance"].items() if v["verdict"].startswith("NOISE")]
    if noisy:
        print("\nmetrics where seed noise exceeds the difference between models:")
        for metric in noisy:
            v = payload["variance"][metric]
            print("  %-22s seeds %.4f vs models %.4f"
                  % (metric, v["spread_between_seeds_worst_case"], v["spread_between_models"]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()