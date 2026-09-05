#!/usr/bin/env python3
"""
peptide_data.py - dataset preparation for the peptide VAE.

Everything here is pure standard library so it can be tested and inspected
without torch. It handles:

  * reading and filtering training.fasta
  * computing the self-derived conditioning properties (length, charge,
    hydrophobic moment, GRAVY, cysteine fraction) from the sequences alone
  * near-duplicate clustering by MinHash over 3-mer sets
  * a cluster-aware train/validation split
  * integer encoding and the empirical distributions used at sampling time

It expects amp_eda.py to sit in the same directory, and reuses the feature
code from it so the VAE and the EDA report always agree on what "charge"
means.
"""

from __future__ import annotations

import json
import math
import os
import random
import zlib
from collections import Counter, defaultdict

try:
    from .amp_eda import (STANDARD_AA, compute_features, longest_run,
                          parse_fasta, quantile)
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "peptide_data needs amp_eda in the same package: %s" % exc
    )

PAD = 0
VOCAB = [""] + list(STANDARD_AA)          # index 0 is the pad token
TOKEN_OF = {aa: i + 1 for i, aa in enumerate(STANDARD_AA)}
VOCAB_SIZE = len(VOCAB)

# The conditioning vector. Every one of these is a function of the sequence
# itself, so no external label file is involved.
PROPERTY_KEYS = ["charge", "hydrophobic_moment", "gravy", "cysteine_frac"]


# ---------------------------------------------------------------------------
# Loading and filtering
# ---------------------------------------------------------------------------

def load_sequences(path, min_len=8, max_len=50, max_run=None,
                   max_residue_frac=None, verbose=True):
    """Read a FASTA file and return the sequences that survive filtering.

    Returns (sequences, rejection_counts).
    """
    records, _ = parse_fasta(path)
    kept = []
    seen = set()
    reasons = Counter()

    for record in records:
        seq = record.seq
        if not seq:
            reasons["empty"] += 1
            continue
        if any(a not in STANDARD_AA for a in seq):
            reasons["non_standard_residue"] += 1
            continue
        if len(seq) < min_len:
            reasons["too_short"] += 1
            continue
        if len(seq) > max_len:
            reasons["too_long"] += 1
            continue
        if max_run is not None and longest_run(seq) > max_run:
            reasons["homopolymer_run"] += 1
            continue
        if max_residue_frac is not None:
            counts = Counter(seq)
            if max(counts.values()) / len(seq) > max_residue_frac:
                reasons["low_complexity"] += 1
                continue
        if seq in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(seq)
        kept.append(seq)

    if verbose:
        print("loaded %d of %d records from %s" % (len(kept), len(records), path))
        for reason, n in reasons.most_common():
            print("  dropped %-22s %d" % (reason, n))
    return kept, reasons


# ---------------------------------------------------------------------------
# Self-derived conditioning properties
# ---------------------------------------------------------------------------

def sequence_properties(sequences, ph=7.0, verbose=True):
    """Compute the conditioning properties for every sequence."""
    rows = []
    for i, seq in enumerate(sequences):
        if verbose and i and i % 10000 == 0:
            print("  properties for %d sequences" % i)
        features = compute_features(seq, ph)
        rows.append([features[key] for key in PROPERTY_KEYS])
    return rows


def standardiser(rows):
    """Return per-property mean and standard deviation for z-scoring."""
    n = len(rows)
    dim = len(PROPERTY_KEYS)
    means = [0.0] * dim
    for row in rows:
        for j in range(dim):
            means[j] += row[j]
    means = [m / max(n, 1) for m in means]
    variances = [0.0] * dim
    for row in rows:
        for j in range(dim):
            variances[j] += (row[j] - means[j]) ** 2
    stds = [math.sqrt(v / max(n - 1, 1)) or 1.0 for v in variances]
    return {"mean": means, "std": stds, "keys": list(PROPERTY_KEYS)}


def standardise(rows, stats):
    mean, std = stats["mean"], stats["std"]
    return [[(row[j] - mean[j]) / std[j] for j in range(len(row))] for row in rows]


# ---------------------------------------------------------------------------
# Near-duplicate clustering (MinHash over 3-mer sets, LSH banding, union-find)
# ---------------------------------------------------------------------------

_MERSENNE = (1 << 61) - 1


def _kmer_set(seq, k):
    if len(seq) < k:
        return {seq}
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def _stable_hash(text):
    """Python's built-in hash() is salted per process, which would make the
    clustering and therefore the train/validation split differ between runs.
    CRC32 is stable across processes and machines."""
    return zlib.crc32(text.encode("ascii"))


def _minhash_signature(kmers, coefficients):
    signature = []
    hashes = [_stable_hash(km) for km in sorted(kmers)]
    for a, b in coefficients:
        best = _MERSENNE
        for h in hashes:
            value = (a * h + b) % _MERSENNE
            if value < best:
                best = value
        signature.append(best)
    return tuple(signature)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cluster_sequences(sequences, k=3, permutations=64, bands=16,
                      threshold=0.5, seed=0, verbose=True):
    """Group near-duplicate sequences. Returns a cluster id per sequence.

    Approximate by design: sequences that land in the same LSH bucket are
    verified against the bucket's first member by true Jaccard similarity of
    their k-mer sets. That is enough to keep peptide families such as the
    indolicidin analogues on one side of a split.
    """
    rng = random.Random(seed)
    rows = permutations // bands
    if rows < 1:
        raise ValueError("permutations must be at least the number of bands")
    coefficients = [(rng.randrange(1, _MERSENNE), rng.randrange(0, _MERSENNE))
                    for _ in range(permutations)]

    kmer_sets = [_kmer_set(seq, k) for seq in sequences]
    if verbose:
        print("hashing %d sequences with %d permutations" % (len(sequences), permutations))
    signatures = [_minhash_signature(km, coefficients) for km in kmer_sets]

    uf = UnionFind(len(sequences))
    for band in range(bands):
        buckets = defaultdict(list)
        lo, hi = band * rows, (band + 1) * rows
        for idx, signature in enumerate(signatures):
            buckets[signature[lo:hi]].append(idx)
        for members in buckets.values():
            if len(members) < 2:
                continue
            anchor = members[0]
            anchor_set = kmer_sets[anchor]
            for other in members[1:]:
                other_set = kmer_sets[other]
                union_size = len(anchor_set | other_set)
                if union_size == 0:
                    continue
                jaccard = len(anchor_set & other_set) / union_size
                if jaccard >= threshold:
                    uf.union(anchor, other)

    roots = {}
    labels = []
    for i in range(len(sequences)):
        root = uf.find(i)
        if root not in roots:
            roots[root] = len(roots)
        labels.append(roots[root])

    if verbose:
        sizes = Counter(labels)
        largest = sizes.most_common(5)
        singletons = sum(1 for c in sizes.values() if c == 1)
        print("found %d clusters, %d singletons, largest %s"
              % (len(sizes), singletons, [n for _, n in largest]))
    return labels


def cluster_aware_split(labels, val_fraction=0.1, seed=0):
    """Split by cluster so no family straddles the train/validation boundary."""
    rng = random.Random(seed)
    by_cluster = defaultdict(list)
    for index, label in enumerate(labels):
        by_cluster[label].append(index)

    clusters = list(by_cluster.values())
    rng.shuffle(clusters)

    target = int(len(labels) * val_fraction)
    val, train = [], []
    for members in clusters:
        if len(val) < target:
            val.extend(members)
        else:
            train.extend(members)
    return sorted(train), sorted(val)


# ---------------------------------------------------------------------------
# Encoding and empirical distributions
# ---------------------------------------------------------------------------

def encode(seq, max_len):
    tokens = [TOKEN_OF[a] for a in seq[:max_len]]
    return tokens + [PAD] * (max_len - len(tokens))


def decode(tokens):
    return "".join(VOCAB[t] for t in tokens if t != PAD)


class EmpiricalSampler:
    """Samples (length, property vector) pairs from the training data.

    Sampling jointly from real training rows rather than independently per
    property keeps the correlations intact: long peptides in this set are not
    the same shape as short ones, and drawing length and charge independently
    would ask the decoder for combinations it never saw.
    """

    def __init__(self, lengths, properties, seed=0):
        self.lengths = list(lengths)
        self.properties = [list(p) for p in properties]
        self.rng = random.Random(seed)

    def sample(self, n):
        picks = [self.rng.randrange(len(self.lengths)) for _ in range(n)]
        return ([self.lengths[i] for i in picks],
                [self.properties[i] for i in picks])

    def sample_with_length(self, n, length):
        """Draw property vectors only from sequences of a matching length."""
        pool = [i for i, l in enumerate(self.lengths) if abs(l - length) <= 2]
        if not pool:
            pool = list(range(len(self.lengths)))
        picks = [self.rng.choice(pool) for _ in range(n)]
        return ([length] * n, [self.properties[i] for i in picks])


# ---------------------------------------------------------------------------
# End to end preparation with caching
# ---------------------------------------------------------------------------

def prepare(path, cache=None, min_len=8, max_len=50, max_run=None,
            max_residue_frac=None, ph=7.0, val_fraction=0.1,
            cluster=True, jaccard=0.5, seed=0, verbose=True):
    """Load, filter, featurise, cluster and split. Cached as JSON."""
    if cache and os.path.exists(cache):
        if verbose:
            print("loading prepared dataset from %s" % cache)
        with open(cache, "r", encoding="utf-8") as handle:
            return json.load(handle)

    sequences, _ = load_sequences(path, min_len, max_len, max_run,
                                  max_residue_frac, verbose)
    if not sequences:
        raise SystemExit("no sequences survived filtering")

    if verbose:
        print("computing self-derived properties")
    raw_properties = sequence_properties(sequences, ph, verbose)
    stats = standardiser(raw_properties)

    if cluster:
        labels = cluster_sequences(sequences, threshold=jaccard, seed=seed,
                                   verbose=verbose)
    else:
        labels = list(range(len(sequences)))

    train_idx, val_idx = cluster_aware_split(labels, val_fraction, seed)
    if verbose:
        print("split: %d train, %d validation" % (len(train_idx), len(val_idx)))

    payload = {
        "path": os.path.abspath(path),
        "max_len": max_len,
        "min_len": min_len,
        "ph": ph,
        "sequences": sequences,
        "lengths": [len(s) for s in sequences],
        "properties_raw": raw_properties,
        "property_stats": stats,
        "cluster_labels": labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        if verbose:
            print("cached prepared dataset to %s" % cache)
    return payload


def length_histogram(lengths, max_len):
    counts = [0] * (max_len + 1)
    for l in lengths:
        if 0 <= l <= max_len:
            counts[l] += 1
    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a peptide dataset.")
    parser.add_argument("fasta")
    parser.add_argument("--cache", default="data/prepared.json")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-run", type=int, default=None,
                        help="drop sequences with a homopolymer run longer than this")
    parser.add_argument("--jaccard", type=float, default=0.5)
    parser.add_argument("--no-cluster", action="store_true")
    args = parser.parse_args()

    data = prepare(args.fasta, cache=args.cache, min_len=args.min_len,
                   max_len=args.max_len, max_run=args.max_run,
                   jaccard=args.jaccard, cluster=not args.no_cluster)
    lengths = data["lengths"]
    ordered = sorted(lengths)
    print("length percentiles: 5%% %d  50%% %d  95%% %d  max %d" % (
        quantile(ordered, 0.05), quantile(ordered, 0.5),
        quantile(ordered, 0.95), ordered[-1]))