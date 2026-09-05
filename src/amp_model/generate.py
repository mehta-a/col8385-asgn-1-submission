"""Generate candidate AMPs, either from the empirical checkpoint or from a trained VAE.

Keeps the round-based loop of the starter kit: sample a batch, keep sequences that are
unique, not present in the exclusion set, and pass the quality filter; if still short of
the target, advance the seed and sample again. Every round's seed is derived from the
top-level --seed, so the whole loop stays deterministic and reruns byte-identical.

Two changes from the starter kit worth knowing about:

  * The exclusion set defaults to training.fasta rather than antibacterial.fasta. The
    antibacterial set is entirely contained in training, so excluding against training
    is a strict superset and needs one fewer input file.
  * The accept predicate does more than deduplicate. The loop was already a rejection
    sampler; it just had nothing to reject on. Low-complexity output is the failure mode
    both the unigram and the VAE share, so it is filtered here.

Determinism notes: numpy draws use a fresh default_rng(seed + round) each round, torch
draws use a fresh manual_seed(seed + round) generator, and PYTHONHASHSEED does not enter
into it because nothing here depends on set or dict iteration order for its output.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

MIN_LENGTH = 8
MAX_LENGTH = 50
BATCH_SIZE = 2_000


# ---------------------------------------------------------------------------
# Input and output
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def _longest_run(seq: str) -> int:
    best = current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best if seq else 0


def make_filter(max_run: int, max_residue_frac: float, min_charge: float | None):
    """Return an accept predicate. Charge is only computed when it is actually used."""

    def accept(seq: str) -> bool:
        if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
            return False
        if max_run and _longest_run(seq) > max_run:
            return False
        if max_residue_frac and max(Counter(seq).values()) / len(seq) > max_residue_frac:
            return False
        if min_charge is not None:
            positive = seq.count("K") + seq.count("R") + 0.1 * seq.count("H")
            if positive - seq.count("D") - seq.count("E") < min_charge:
                return False
        return True

    return accept


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class EmpiricalSampler:
    """The starter kit baseline: independent residues, empirical length distribution."""

    def __init__(self, checkpoint: dict):
        letters, freqs = zip(*checkpoint["aa_freqs"].items())
        self.letters = np.asarray(letters)
        self.aa_p = np.asarray(freqs, dtype=float)
        self.aa_p /= self.aa_p.sum()

        lengths, probs = zip(*checkpoint["length_probs"].items())
        self.lengths = np.asarray([int(length) for length in lengths])
        self.length_p = np.asarray(probs, dtype=float)
        self.length_p /= self.length_p.sum()

    def batch(self, batch_size: int, seed: int) -> list[str]:
        rng = np.random.default_rng(seed)
        sampled = np.clip(rng.choice(self.lengths, size=batch_size, p=self.length_p),
                          MIN_LENGTH, MAX_LENGTH)
        # One vectorised draw for every residue in the batch, then split by length.
        total = int(sampled.sum())
        residues = rng.choice(self.letters, size=total, p=self.aa_p)
        sequences = []
        cursor = 0
        for length in sampled:
            sequences.append("".join(residues[cursor:cursor + length]))
            cursor += length
        return sequences


class VAESampler:
    """Samples from a trained conditional VAE checkpoint."""

    def __init__(self, checkpoint_path: Path, temperature: float, refine: int,
                 prior: str, device: str = "cpu"):
        import torch

        from . import peptide_data as pdata
        from .peptide_vae import draw_latents, load_model, sample_tokens

        self.torch = torch
        self.pdata = pdata
        self.draw_latents = draw_latents
        self.sample_tokens = sample_tokens

        self.device = torch.device(device)
        self.model, self.blob = load_model(str(checkpoint_path), self.device)
        self.temperature = temperature
        self.refine = refine
        self.prior = prior
        self.stats = self.blob["property_stats"]
        self.empirical_lengths = self.blob["empirical"]["lengths"]
        self.empirical_properties = self.blob["empirical"]["properties"]

    def batch(self, batch_size: int, seed: int) -> list[str]:
        torch = self.torch
        torch.manual_seed(seed)

        # Draw (length, property) pairs jointly from real training rows so the
        # correlations between them survive.
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(self.empirical_lengths), size=batch_size)
        lengths_list = [self.empirical_lengths[i] for i in picks]
        properties_list = [self.empirical_properties[i] for i in picks]

        lengths = torch.tensor(lengths_list, dtype=torch.long, device=self.device)
        properties = torch.tensor(
            self.pdata.standardise(properties_list, self.stats),
            dtype=torch.float32, device=self.device,
        )
        z = self.draw_latents(self.blob, batch_size, self.device, self.prior)

        sequences = []
        with torch.no_grad():
            for start in range(0, batch_size, 512):
                end = min(start + 512, batch_size)
                batch_lengths = lengths[start:end]
                batch_properties = properties[start:end]

                condition = self.model.conditioner(batch_lengths, batch_properties)
                mask = self.model.make_mask(batch_lengths)
                tokens = self.sample_tokens(self.model, z[start:end], condition, mask,
                                            self.temperature)
                for _ in range(self.refine):
                    mu, _, condition, mask = self.model.encode(tokens, batch_lengths,
                                                               batch_properties)
                    tokens = self.sample_tokens(self.model, mu, condition, mask,
                                                self.temperature)

                for row, length in zip(tokens.cpu().tolist(), batch_lengths.cpu().tolist()):
                    sequences.append(self.pdata.decode(row[:length]))
        return sequences


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def generate(
    n_sequences: int,
    sampler,
    excluded: set[str],
    accept,
    *,
    seed: int = 42,
    batch_size: int = BATCH_SIZE,
    max_rounds: int = 200,
) -> tuple[list[str], dict]:
    collected: list[str] = []
    seen: set[str] = set()
    stats = Counter()
    round_idx = 0

    while len(collected) < n_sequences:
        if round_idx >= max_rounds:
            raise SystemExit(
                f"stopped after {max_rounds} rounds with {len(collected)} of {n_sequences} "
                f"sequences. Rejections so far: {dict(stats)}. Loosen the filter or raise "
                f"the temperature."
            )
        for seq in sampler.batch(batch_size, seed + round_idx):
            if len(collected) >= n_sequences:
                break
            if seq in seen:
                stats["duplicate_of_generated"] += 1
                continue
            if seq in excluded:
                stats["present_in_training"] += 1
                continue
            if not accept(seq):
                stats["failed_filter"] += 1
                continue
            seen.add(seq)
            collected.append(seq)
        round_idx += 1

    stats["rounds"] = round_idx
    return collected, dict(stats)


def resolve_model_path(explicit: Path | None, checkpoint: Path) -> Path | None:
    """Decide whether to sample from a VAE checkpoint or the empirical one.

    The grader invokes `generate` with no arguments at all, so this has to work
    out what the committed checkpoint/ directory actually holds:

      1. an explicit --model always wins
      2. a model.pt sitting beside the JSON
      3. a "weights" pointer inside the JSON, resolved relative to the JSON
      4. otherwise the JSON is an empirical checkpoint, so return None
    """
    if explicit:
        return explicit
    if not checkpoint.exists():
        return None

    sibling = checkpoint.parent / "model.pt"
    if sibling.exists():
        return sibling

    try:
        blob = json.loads(checkpoint.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if "aa_freqs" in blob:
        return None

    pointer = blob.get("weights")
    if pointer:
        for candidate in (checkpoint.parent / Path(pointer).name, Path(pointer)):
            if candidate.exists():
                return candidate
        raise SystemExit(
            f"{checkpoint} describes a VAE whose weights are at {pointer}, but that file "
            f"is missing. Commit model.pt next to {checkpoint.name}."
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sequences", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/model.json"),
                        help="empirical checkpoint, used when --model is not given")
    parser.add_argument("--model", type=Path, default=None,
                        help="trained VAE checkpoint (.pt); overrides --checkpoint")
    parser.add_argument("--exclude-fasta", type=Path,
                        default=Path("data/training/training.fasta"),
                        help="sequences never to emit (default: the training set)")
    parser.add_argument("--out", type=Path, default=Path("generate/library.fasta"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--prior", choices=["fitted", "normal"], default="fitted")
    parser.add_argument("--max-run", type=int, default=5,
                        help="reject homopolymer runs longer than this (0 disables)")
    parser.add_argument("--max-residue-frac", type=float, default=0.5,
                        help="reject sequences more than this fraction of one residue")
    parser.add_argument("--min-charge", type=float, default=None,
                        help="reject below this crude net charge (off by default, since "
                             "it distorts the charge distribution)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    excluded = (
        set(_read_fasta_sequences(args.exclude_fasta))
        if args.exclude_fasta and args.exclude_fasta.exists()
        else set()
    )
    if not excluded:
        print(f"warning: no exclusion set loaded from {args.exclude_fasta}")

    model_path = resolve_model_path(args.model, args.checkpoint)
    if model_path:
        sampler = VAESampler(model_path, args.temperature, args.refine, args.prior, args.device)
        source = f"VAE {model_path}"
    else:
        checkpoint = json.loads(args.checkpoint.read_text())
        sampler = EmpiricalSampler(checkpoint)
        source = f"empirical {args.checkpoint}"

    accept = make_filter(args.max_run, args.max_residue_frac, args.min_charge)
    sequences, stats = generate(
        args.n_sequences, sampler, excluded, accept,
        seed=args.seed, batch_size=args.batch_size,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write_fasta(sequences, args.out)
    print(f"Generated {len(sequences)} sequences from {source} -> {args.out}")
    print(f"  rejected: {stats}")


if __name__ == "__main__":
    main()