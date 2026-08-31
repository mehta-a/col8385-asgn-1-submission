"""Generate 1,000 candidate AMPs from the empirical checkpoint produced by train.py.

Uses the same round-based loop as ampdiffusion-starter-kit's generate_library(): sample a batch,
keep sequences that are unique and not already present in antibacterial.fasta, and if still short
of the target, sample another batch with the next seed in sequence and repeat. Each round's seed
is fully determined by the top-level --seed, so the loop is itself deterministic -- this is what
lets the byte-identical rerun check pass without any special-casing.
"""

import argparse
import json
from pathlib import Path

import numpy as np

MIN_LENGTH = 8
MAX_LENGTH = 50
BATCH_SIZE = 2_000


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


def _write_fasta(sequences: list[str], path: Path) -> None:
    with open(path, "w") as f:
        for i, seq in enumerate(sequences, start=1):
            f.write(f">seq{i}\n{seq}\n")


def _sample_batch(checkpoint: dict, batch_size: int, rng: np.random.Generator) -> list[str]:
    aa_letters, aa_freqs = zip(*checkpoint["aa_freqs"].items())
    aa_p = np.asarray(aa_freqs, dtype=float)
    aa_p /= aa_p.sum()

    lengths = [int(length) for length in checkpoint["length_probs"]]
    length_p = np.asarray(list(checkpoint["length_probs"].values()), dtype=float)
    length_p /= length_p.sum()

    sampled_lengths = rng.choice(lengths, size=batch_size, p=length_p)
    sequences = []
    for length in sampled_lengths:
        length = int(np.clip(length, MIN_LENGTH, MAX_LENGTH))
        sequences.append("".join(rng.choice(aa_letters, size=length, p=aa_p)))
    return sequences


def generate(
    n_sequences: int,
    checkpoint: dict,
    antibacterial_sequences: set[str],
    *,
    seed: int = 42,
    batch_size: int = BATCH_SIZE,
) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    round_idx = 0
    while len(collected) < n_sequences:
        rng = np.random.default_rng(seed + round_idx)
        for seq in _sample_batch(checkpoint, batch_size, rng):
            if len(collected) >= n_sequences:
                break
            if seq in seen or seq in antibacterial_sequences:
                continue
            seen.add(seq)
            collected.append(seq)
        round_idx += 1
    return collected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sequences", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/model.json"))
    parser.add_argument("--antibacterial-fasta", type=Path, default=Path("data/antibacterial.fasta"))
    args = parser.parse_args()

    checkpoint = json.loads(args.checkpoint.read_text())
    antibacterial_sequences = (
        set(_read_fasta_sequences(args.antibacterial_fasta)) if args.antibacterial_fasta.exists() else set()
    )

    sequences = generate(args.n_sequences, checkpoint, antibacterial_sequences, seed=args.seed)

    out_dir = Path("generate")
    out_dir.mkdir(parents=True, exist_ok=True)
    library_path = out_dir / "library.fasta"
    _write_fasta(sequences, library_path)
    print(f"Generated {len(sequences)} sequences -> {library_path}")


if __name__ == "__main__":
    main()