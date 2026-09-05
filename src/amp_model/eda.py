#!/usr/bin/env python3
"""
amp_eda.py - exploratory data analysis for peptide FASTA datasets.

Reads one or more FASTA files, computes sequence-level and dataset-level
statistics, and writes a single self-contained HTML report (inline SVG, no
external assets, no network access required to view it).

Standard library only. No numpy, no biopython, no matplotlib.

Usage
-----
    python amp_eda.py training.fasta antibacteria.fasta background.fasta
    python amp_eda.py *.fasta --out reports/eda.html
    python amp_eda.py a.fasta b.fasta --positive a --negative b --ph 7.4

Labels default to the file stem. Pass name=path to override:

    python amp_eda.py train=data/training.fasta neg=data/background.fasta
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime

# ---------------------------------------------------------------------------
# Section 1. Amino acid constants
# ---------------------------------------------------------------------------

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# Ambiguity / non-standard codes that may legitimately appear in a FASTA file.
AMBIGUOUS_AA = {
    "X": "any residue",
    "B": "asparagine or aspartate",
    "Z": "glutamine or glutamate",
    "J": "leucine or isoleucine",
    "U": "selenocysteine",
    "O": "pyrrolysine",
}

# Kyte-Doolittle hydropathy. Used for GRAVY.
KD_HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Eisenberg consensus scale. Used for the hydrophobic moment.
EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

# Average residue masses in daltons. Add one water for the free peptide.
RESIDUE_MASS = {
    "G": 57.0519, "A": 71.0788, "S": 87.0782, "P": 97.1167, "V": 99.1326,
    "T": 101.1051, "C": 103.1388, "L": 113.1594, "I": 113.1594, "N": 114.1038,
    "D": 115.0886, "Q": 128.1307, "K": 128.1741, "E": 129.1155, "M": 131.1926,
    "H": 137.1411, "F": 147.1766, "R": 156.1875, "Y": 163.1760, "W": 186.2132,
}
WATER_MASS = 18.0153

# EMBOSS pKa set, used for net charge and isoelectric point.
PKA_NTERM = 8.6
PKA_CTERM = 3.6
PKA_POSITIVE = {"K": 10.8, "R": 12.5, "H": 6.5}
PKA_NEGATIVE = {"D": 3.9, "E": 4.1, "C": 8.5, "Y": 10.1}

# Residue classes. These drive the colour scheme of the whole report, so the
# same residue reads the same way in every figure.
RESIDUE_CLASS = {}
for _aa in "AVLIMFW":
    RESIDUE_CLASS[_aa] = "hydrophobic"
for _aa in "STNQ":
    RESIDUE_CLASS[_aa] = "polar"
for _aa in "KR":
    RESIDUE_CLASS[_aa] = "basic"
for _aa in "DE":
    RESIDUE_CLASS[_aa] = "acidic"
RESIDUE_CLASS["H"] = "aromatic"
RESIDUE_CLASS["Y"] = "aromatic"
RESIDUE_CLASS["G"] = "glycine"
RESIDUE_CLASS["P"] = "proline"
RESIDUE_CLASS["C"] = "cysteine"

CLASS_COLOUR = {
    "hydrophobic": "#3b7dd8",
    "polar": "#2e9e5b",
    "basic": "#d6403a",
    "acidic": "#b44bc8",
    "aromatic": "#1f9aa3",
    "glycine": "#e0891e",
    "proline": "#b8902a",
    "cysteine": "#dd6f9f",
    "other": "#8a938f",
}

CLASS_LABEL = OrderedDict([
    ("hydrophobic", "Hydrophobic (A V L I M F W)"),
    ("polar", "Polar (S T N Q)"),
    ("basic", "Basic (K R)"),
    ("acidic", "Acidic (D E)"),
    ("aromatic", "Aromatic ring (H Y)"),
    ("glycine", "Glycine"),
    ("proline", "Proline"),
    ("cysteine", "Cysteine"),
])

# Series colours for per-dataset overlays. Deliberately outside the residue
# palette so the two encodings never get confused.
SERIES_COLOURS = ["#16202a", "#17786f", "#a8641a", "#6a4fa3", "#8c1c4d", "#3f6d20"]

FEATURE_ORDER = [
    ("length", "Length", "residues"),
    ("charge", "Net charge", "e at pH {ph}"),
    ("charge_density", "Charge per residue", "e/residue"),
    ("gravy", "GRAVY", "Kyte-Doolittle mean"),
    ("hydrophobic_moment", "Hydrophobic moment", "Eisenberg, window 11"),
    ("hydrophobic_frac", "Hydrophobic fraction", "A V L I M F W"),
    ("aromatic_frac", "Aromatic fraction", "F W Y H"),
    ("basic_frac", "Basic fraction", "K R H"),
    ("acidic_frac", "Acidic fraction", "D E"),
    ("cysteine_frac", "Cysteine fraction", "C"),
    ("glycine_frac", "Glycine fraction", "G"),
    ("proline_frac", "Proline fraction", "P"),
    ("aliphatic_index", "Aliphatic index", "thermostability proxy"),
    ("isoelectric_point", "Isoelectric point", "pI"),
    ("molecular_weight", "Molecular weight", "Da"),
    ("entropy", "Sequence entropy", "bits, max 4.32"),
    ("max_run", "Longest identical run", "residues"),
    ("max_residue_frac", "Most frequent residue share", "fraction"),
]


# ---------------------------------------------------------------------------
# Section 2. FASTA parsing
# ---------------------------------------------------------------------------

class Record:
    __slots__ = ("header", "seq", "line_no")

    def __init__(self, header, seq, line_no):
        self.header = header
        self.seq = seq
        self.line_no = line_no


def parse_fasta(path):
    """Parse a FASTA file into records plus a list of parse warnings.

    Sequences are upper-cased. Gap characters (-, .) and trailing stop codons
    (*) are stripped, and each removal is reported.
    """
    records = []
    warnings = []
    header = None
    chunks = []
    start_line = 0
    saw_any_line = False

    def flush():
        if header is None:
            return
        raw = "".join(chunks)
        cleaned = raw.upper()
        stripped = re.sub(r"[\-\.\*\s]", "", cleaned)
        if len(stripped) != len(cleaned):
            warnings.append(
                "line %d: removed %d gap, stop or whitespace characters from %s"
                % (start_line, len(cleaned) - len(stripped), short_header(header))
            )
        if not stripped:
            warnings.append("line %d: empty sequence for %s" % (start_line, short_header(header)))
        records.append(Record(header, stripped, start_line))

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if not saw_any_line:
                saw_any_line = True
                if not line.startswith(">"):
                    warnings.append("line 1: file does not start with a header line")
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                chunks = []
                start_line = line_no
            else:
                if header is None:
                    header = "(no header)"
                    start_line = line_no
                chunks.append(line.strip())
    flush()
    return records, warnings


def short_header(header):
    header = header or "(no header)"
    return header if len(header) <= 40 else header[:37] + "..."


# ---------------------------------------------------------------------------
# Section 3. Per-sequence features
# ---------------------------------------------------------------------------

def net_charge(counts, length, ph):
    """Net charge from Henderson-Hasselbalch over ionisable groups."""
    if length == 0:
        return 0.0
    positive = 1.0 / (1.0 + 10 ** (ph - PKA_NTERM))
    for aa, pka in PKA_POSITIVE.items():
        n = counts.get(aa, 0)
        if n:
            positive += n / (1.0 + 10 ** (ph - pka))
    negative = 1.0 / (1.0 + 10 ** (PKA_CTERM - ph))
    for aa, pka in PKA_NEGATIVE.items():
        n = counts.get(aa, 0)
        if n:
            negative += n / (1.0 + 10 ** (pka - ph))
    return positive - negative


def isoelectric_point(counts, length):
    """Bisection on the charge curve between pH 0 and 14."""
    if length == 0:
        return 0.0
    low, high = 0.0, 14.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if net_charge(counts, length, mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def hydrophobic_moment(seq, window=11, angle_deg=100.0):
    """Maximum Eisenberg hydrophobic moment over sliding windows."""
    n = len(seq)
    if n == 0:
        return 0.0
    w = min(window, n)
    angle = math.radians(angle_deg)
    best = 0.0
    for start in range(0, n - w + 1):
        sin_sum = 0.0
        cos_sum = 0.0
        for i in range(w):
            h = EISENBERG.get(seq[start + i], 0.0)
            sin_sum += h * math.sin(angle * i)
            cos_sum += h * math.cos(angle * i)
        best = max(best, math.sqrt(sin_sum * sin_sum + cos_sum * cos_sum) / w)
    return best


def shannon_entropy(counts, length):
    if length == 0:
        return 0.0
    total = 0.0
    for n in counts.values():
        if n:
            p = n / length
            total -= p * math.log2(p)
    return total


def longest_run(seq):
    if not seq:
        return 0
    best = 1
    current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            if current > best:
                best = current
        else:
            current = 1
    return best


def compute_features(seq, ph):
    counts = Counter(seq)
    n = len(seq)
    frac = (lambda group: sum(counts.get(a, 0) for a in group) / n) if n else (lambda group: 0.0)

    mass = sum(RESIDUE_MASS.get(a, 0.0) * c for a, c in counts.items()) + (WATER_MASS if n else 0.0)
    gravy = (sum(KD_HYDROPATHY.get(a, 0.0) * c for a, c in counts.items()) / n) if n else 0.0
    charge = net_charge(counts, n, ph)

    aliphatic = 0.0
    if n:
        a = counts.get("A", 0) / n * 100.0
        v = counts.get("V", 0) / n * 100.0
        il = (counts.get("I", 0) + counts.get("L", 0)) / n * 100.0
        aliphatic = a + 2.9 * v + 3.9 * il

    return {
        "length": float(n),
        "charge": charge,
        "charge_density": charge / n if n else 0.0,
        "gravy": gravy,
        "hydrophobic_moment": hydrophobic_moment(seq),
        "hydrophobic_frac": frac("AVLIMFW"),
        "aromatic_frac": frac("FWYH"),
        "basic_frac": frac("KRH"),
        "acidic_frac": frac("DE"),
        "cysteine_frac": frac("C"),
        "glycine_frac": frac("G"),
        "proline_frac": frac("P"),
        "aliphatic_index": aliphatic,
        "isoelectric_point": isoelectric_point(counts, n),
        "molecular_weight": mass,
        "entropy": shannon_entropy(counts, n),
        "max_run": float(longest_run(seq)),
        "max_residue_frac": (max(counts.values()) / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Section 4. Statistics helpers
# ---------------------------------------------------------------------------

def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(sorted_values) - 1)
    weight = pos - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def describe(values):
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": mean(ordered),
        "sd": stdev(ordered),
        "min": ordered[0] if ordered else 0.0,
        "p05": quantile(ordered, 0.05),
        "p25": quantile(ordered, 0.25),
        "median": quantile(ordered, 0.50),
        "p75": quantile(ordered, 0.75),
        "p95": quantile(ordered, 0.95),
        "max": ordered[-1] if ordered else 0.0,
    }


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sa, sb = stdev(a), stdev(b)
    pooled_var = ((na - 1) * sa * sa + (nb - 1) * sb * sb) / (na + nb - 2)
    if pooled_var <= 0:
        return 0.0
    return (mean(a) - mean(b)) / math.sqrt(pooled_var)


def auroc(positives, negatives):
    """Rank-based AUROC with tie correction (Mann-Whitney U / n1 n2)."""
    n1, n2 = len(positives), len(negatives)
    if n1 == 0 or n2 == 0:
        return 0.5
    tagged = [(v, 1) for v in positives] + [(v, 0) for v in negatives]
    tagged.sort(key=lambda t: t[0])
    ranks = [0.0] * len(tagged)
    i = 0
    while i < len(tagged):
        j = i
        while j + 1 < len(tagged) and tagged[j + 1][0] == tagged[i][0]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    rank_sum = sum(ranks[k] for k in range(len(tagged)) if tagged[k][1] == 1)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def histogram(values, low, high, bins):
    counts = [0] * bins
    if high <= low:
        high = low + 1.0
    width = (high - low) / bins
    for v in values:
        if v < low or v > high:
            continue
        idx = int((v - low) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts, low, high


def kmer_counts(sequences, k):
    counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            counts[seq[i:i + k]] += 1
    return counts


# ---------------------------------------------------------------------------
# Section 5. Dataset container
# ---------------------------------------------------------------------------

class Dataset:
    def __init__(self, label, path, ph, kmer_k):
        self.label = label
        self.path = path
        self.ph = ph
        self.records, self.warnings = parse_fasta(path)
        self.sequences = [r.seq for r in self.records if r.seq]
        self.n_records = len(self.records)
        self.n_empty = sum(1 for r in self.records if not r.seq)

        self.file_size = os.path.getsize(path) if os.path.exists(path) else 0

        seen = Counter(self.sequences)
        self.unique_sequences = set(seen)
        self.n_duplicate_rows = sum(c - 1 for c in seen.values() if c > 1)
        self.top_duplicates = seen.most_common(5)

        header_seen = Counter(r.header for r in self.records)
        self.n_duplicate_headers = sum(c - 1 for c in header_seen.values() if c > 1)
        self.sample_headers = [r.header for r in self.records[:3]]

        self.residue_counts = Counter()
        for seq in self.sequences:
            self.residue_counts.update(seq)
        self.total_residues = sum(self.residue_counts.values())
        self.nonstandard = {a: c for a, c in self.residue_counts.items() if a not in STANDARD_AA}
        self.n_seq_with_nonstandard = sum(
            1 for seq in self.sequences if any(a not in STANDARD_AA for a in seq)
        )

        self.features = {key: [] for key, _, _ in FEATURE_ORDER}
        for seq in self.sequences:
            values = compute_features(seq, ph)
            for key in self.features:
                self.features[key].append(values[key])

        self.summary = {key: describe(vals) for key, vals in self.features.items()}

        self.nterm = Counter(seq[0] for seq in self.sequences)
        self.cterm = Counter(seq[-1] for seq in self.sequences)
        self.kmers = kmer_counts(self.sequences, kmer_k)
        self.kmer_total = sum(self.kmers.values())

    def composition(self):
        """Fraction of each standard residue over all residues in the file."""
        if not self.total_residues:
            return {a: 0.0 for a in STANDARD_AA}
        return {a: self.residue_counts.get(a, 0) / self.total_residues for a in STANDARD_AA}

    def extremes(self):
        out = []
        if not self.sequences:
            return out
        pairs = list(zip(self.sequences, self.features["length"]))
        longest = max(pairs, key=lambda t: t[1])[0]
        shortest = min(pairs, key=lambda t: t[1])[0]
        charged = max(zip(self.sequences, self.features["charge"]), key=lambda t: t[1])
        hydrophobic = max(zip(self.sequences, self.features["gravy"]), key=lambda t: t[1])
        out.append(("Longest", longest, "%d residues" % len(longest)))
        out.append(("Shortest", shortest, "%d residues" % len(shortest)))
        out.append(("Most positively charged", charged[0], "%+.1f e" % charged[1]))
        out.append(("Most hydrophobic", hydrophobic[0], "GRAVY %.2f" % hydrophobic[1]))
        return out


# ---------------------------------------------------------------------------
# Section 6. SVG drawing helpers
# ---------------------------------------------------------------------------

def esc(text):
    return html.escape(str(text), quote=True)


def fmt(value, digits=2):
    if isinstance(value, int):
        return "{:,}".format(value)
    if value != value:  # NaN
        return "n/a"
    if abs(value) >= 10000:
        return "{:,.0f}".format(value)
    return ("{:,.%df}" % digits).format(value)


def plural(count, singular, suffix="s"):
    return "%s %s%s" % (fmt(int(count)), singular, "" if count == 1 else suffix)


def axis_ticks(low, high, count=5):
    """Readable tick positions across a range."""
    if high <= low:
        return [low]
    span = high - low
    raw_step = span / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if span / step <= count + 1:
            break
    start = math.ceil(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks or [low, high]


def tick_label(value):
    if abs(value) >= 1000:
        return "{:,.0f}".format(value)
    if abs(value) >= 10:
        return "{:.0f}".format(value)
    if abs(value) >= 1:
        return "{:.1f}".format(value)
    return "{:.2f}".format(value)


def svg_distribution(series, title, unit, bins=44, width=760, height=210):
    """Overlaid density outlines, one polyline per dataset.

    Each series is normalised to a fraction of its own sequences per bin, so
    files of very different sizes stay comparable.
    """
    all_values = [v for values in series.values() for v in values]
    if not all_values:
        return "<p class='empty'>No data for %s.</p>" % esc(title)

    pooled = sorted(all_values)
    low = quantile(pooled, 0.002)
    high = quantile(pooled, 0.998)
    if high <= low:
        low, high = pooled[0], pooled[0] + 1.0

    left, right, top, bottom = 46, 12, 16, 34
    plot_w = width - left - right
    plot_h = height - top - bottom

    densities = {}
    peak = 0.0
    for label, values in series.items():
        counts, _, _ = histogram(values, low, high, bins)
        total = max(len(values), 1)
        dens = [c / total for c in counts]
        densities[label] = dens
        peak = max(peak, max(dens) if dens else 0.0)
    if peak <= 0:
        peak = 1.0

    def x_of(value):
        return left + (value - low) / (high - low) * plot_w

    def y_of(density):
        return top + plot_h - (density / peak) * plot_h

    parts = ['<svg class="chart" viewBox="0 0 %d %d" role="img" aria-label="%s distribution">'
             % (width, height, esc(title))]

    for tick in axis_ticks(low, high, 6):
        x = x_of(tick)
        parts.append('<line class="grid" x1="%.1f" y1="%d" x2="%.1f" y2="%.1f"/>'
                     % (x, top, x, top + plot_h))
        parts.append('<text class="tick" x="%.1f" y="%.1f" text-anchor="middle">%s</text>'
                     % (x, top + plot_h + 15, esc(tick_label(tick))))
    parts.append('<line class="axis" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>'
                 % (left, top + plot_h, width - right, top + plot_h))
    parts.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%s</text>'
                 % (left - 6, top + 5, esc(tick_label(peak * 100)) + "%"))
    parts.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">0</text>'
                 % (left - 6, top + plot_h))

    step = plot_w / bins
    for index, (label, dens) in enumerate(densities.items()):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        points = ["%.2f,%.2f" % (left, top + plot_h)]
        for b, d in enumerate(dens):
            x0 = left + b * step
            x1 = left + (b + 1) * step
            y = y_of(d)
            points.append("%.2f,%.2f" % (x0, y))
            points.append("%.2f,%.2f" % (x1, y))
        points.append("%.2f,%.2f" % (left + plot_w, top + plot_h))
        path = " ".join(points)
        parts.append('<polygon class="dens-fill" data-series="%s" points="%s" fill="%s"/>'
                     % (esc(label), path, colour))
        parts.append('<polyline class="dens-line" data-series="%s" points="%s" stroke="%s"/>'
                     % (esc(label), path, colour))

    parts.append('<text class="axis-label" x="%.1f" y="%d" text-anchor="middle">%s</text>'
                 % (left + plot_w / 2, height - 4, esc(unit)))
    parts.append("</svg>")
    return "".join(parts)


def svg_composition(comp, label, width=760, row_height=132):
    """Vertical bars for the 20 standard residues, coloured by residue class."""
    order = sorted(STANDARD_AA, key=lambda a: (list(CLASS_LABEL).index(RESIDUE_CLASS[a]), a))
    left, right, top, bottom = 34, 8, 12, 26
    plot_w = width - left - right
    plot_h = row_height - top - bottom
    peak = max(comp.values()) if comp else 0.1
    peak = max(peak, 0.01)
    slot = plot_w / len(order)
    bar_w = slot * 0.62

    parts = ['<svg class="chart" viewBox="0 0 %d %d" role="img" aria-label="Residue composition of %s">'
             % (width, row_height, esc(label))]
    for tick in (0.0, peak / 2, peak):
        y = top + plot_h - (tick / peak) * plot_h
        parts.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>'
                     % (left, y, width - right, y))
        parts.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%s</text>'
                     % (left - 5, y + 3, esc("%.0f%%" % (tick * 100))))
    for i, aa in enumerate(order):
        value = comp.get(aa, 0.0)
        h = (value / peak) * plot_h
        x = left + i * slot + (slot - bar_w) / 2
        colour = CLASS_COLOUR[RESIDUE_CLASS[aa]]
        parts.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s" rx="1">'
                     '<title>%s: %.2f%%</title></rect>'
                     % (x, top + plot_h - h, bar_w, max(h, 0.4), colour, aa, value * 100))
        parts.append('<text class="residue" x="%.2f" y="%.1f" text-anchor="middle" fill="%s">%s</text>'
                     % (x + bar_w / 2, top + plot_h + 16, colour, aa))
    parts.append("</svg>")
    return "".join(parts)


def svg_enrichment(values, width=760, height=190, unit="log2 ratio"):
    """Diverging horizontal-zero bar chart over the 20 residues."""
    order = sorted(STANDARD_AA, key=lambda a: (list(CLASS_LABEL).index(RESIDUE_CLASS[a]), a))
    left, right, top, bottom = 44, 8, 14, 30
    plot_w = width - left - right
    plot_h = height - top - bottom
    limit = max((abs(values.get(a, 0.0)) for a in order), default=1.0)
    limit = max(limit, 0.1)
    zero_y = top + plot_h / 2
    slot = plot_w / len(order)
    bar_w = slot * 0.62

    parts = ['<svg class="chart" viewBox="0 0 %d %d" role="img" aria-label="Residue enrichment">'
             % (width, height)]
    for tick in (-limit, -limit / 2, 0.0, limit / 2, limit):
        y = zero_y - (tick / limit) * (plot_h / 2)
        parts.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>'
                     % (left, y, width - right, y))
        parts.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%s</text>'
                     % (left - 5, y + 3, esc("%+.1f" % tick if tick else "0")))
    for i, aa in enumerate(order):
        value = values.get(aa, 0.0)
        h = abs(value) / limit * (plot_h / 2)
        x = left + i * slot + (slot - bar_w) / 2
        y = zero_y - h if value >= 0 else zero_y
        colour = CLASS_COLOUR[RESIDUE_CLASS[aa]]
        parts.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s" rx="1">'
                     '<title>%s: %+.2f</title></rect>'
                     % (x, y, bar_w, max(h, 0.4), colour, aa, value))
        label_y = zero_y + 15 if value >= 0 else zero_y - 6
        parts.append('<text class="residue" x="%.2f" y="%.1f" text-anchor="middle" fill="%s">%s</text>'
                     % (x + bar_w / 2, label_y, colour, aa))
    parts.append('<text class="axis-label" x="%.1f" y="%d" text-anchor="middle">%s</text>'
                 % (left + plot_w / 2, height - 4, esc(unit)))
    parts.append("</svg>")
    return "".join(parts)


def svg_separability(rows, width=760, bar_height=20):
    """Horizontal AUROC bars measured from the 0.5 no-information line."""
    if not rows:
        return ""
    left, right, top, bottom = 176, 46, 20, 26
    plot_w = width - left - right
    height = top + bottom + bar_height * len(rows)
    low, high = 0.0, 1.0

    def x_of(value):
        return left + (value - low) / (high - low) * plot_w

    parts = ['<svg class="chart" viewBox="0 0 %d %d" role="img" aria-label="Single feature separability">'
             % (width, height)]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x_of(tick)
        css = "axis" if tick == 0.5 else "grid"
        parts.append('<line class="%s" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>'
                     % (css, x, top - 6, x, height - bottom))
        parts.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>'
                     % (x, height - bottom + 15, esc("%.2f" % tick)))
    for i, (name, value, d) in enumerate(rows):
        y = top + i * bar_height
        x0 = x_of(min(value, 0.5))
        x1 = x_of(max(value, 0.5))
        colour = "#16202a" if abs(value - 0.5) > 0.15 else "#8a938f"
        parts.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%d" fill="%s" rx="1"/>'
                     % (x0, y + 3, max(x1 - x0, 0.6), bar_height - 8, colour))
        parts.append('<text class="row-label" x="%d" y="%.1f" text-anchor="end">%s</text>'
                     % (left - 10, y + bar_height - 6, esc(name)))
        parts.append('<text class="row-value" x="%d" y="%.1f">%s</text>'
                     % (width - right + 6, y + bar_height - 6, esc("%.3f" % value)))
    parts.append("</svg>")
    return "".join(parts)


def sequence_band(datasets, max_len=68):
    """Hero element: real sequences from the data, coloured by residue class."""
    rows = []
    for ds in datasets:
        if not ds.sequences:
            continue
        pick = min(ds.sequences, key=lambda s: abs(len(s) - 40))
        letters = pick[:max_len]
        spans = "".join(
            '<span class="r-%s">%s</span>' % (RESIDUE_CLASS.get(a, "other"), esc(a))
            for a in letters
        )
        if len(pick) > max_len:
            spans += '<span class="r-other">...</span>'
        rows.append('<div class="band-row"><span class="band-name">%s</span>'
                    '<span class="band-seq">%s</span></div>' % (esc(ds.label), spans))
    return "".join(rows)


# ---------------------------------------------------------------------------
# Section 7. HTML report
# ---------------------------------------------------------------------------

CSS = """
:root {
  --paper: #f6f7f5;
  --panel: #ffffff;
  --ink: #16202a;
  --ink-soft: #55605c;
  --rule: #dde2dd;
  --accent: #17786f;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0 20px 96px;
  background: var(--paper);
  color: var(--ink);
  font-family: "Avenir Next", Avenir, "Segoe UI", system-ui, sans-serif;
  font-size: 16px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 860px; margin: 0 auto; }
h1 { font-size: 30px; font-weight: 600; letter-spacing: -0.01em; margin: 44px 0 6px; }
h2 { font-size: 21px; font-weight: 600; margin: 56px 0 4px; }
h3 { font-size: 15px; font-weight: 600; margin: 26px 0 6px; color: var(--ink-soft); }
p { max-width: 68ch; margin: 8px 0 14px; }
p.note { color: var(--ink-soft); font-size: 14px; }
a { color: var(--accent); }
.meta { color: var(--ink-soft); font-size: 14px; margin-bottom: 26px; }
.hero {
  background: var(--panel);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: 18px 20px;
  overflow-x: auto;
}
.band-row { display: flex; gap: 16px; align-items: baseline; padding: 3px 0; white-space: nowrap; }
.band-name {
  min-width: 118px; text-align: right; font-size: 12px; color: var(--ink-soft);
  flex: 0 0 auto;
}
.band-seq { font-family: Menlo, "SF Mono", Consolas, monospace; font-size: 15px; letter-spacing: 0.06em; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 12px 0 0; font-size: 12.5px; color: var(--ink-soft); }
.legend span::before {
  content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 2px;
  margin-right: 6px; background: currentColor;
}
table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 10px 0 6px; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; }
thead th { color: var(--ink-soft); font-weight: 500; border-bottom: 1px solid var(--ink); }
tbody tr:hover { background: #eef1ee; }
td.num, th.num { font-family: Menlo, "SF Mono", Consolas, monospace; font-size: 13px; }
td.seq { font-family: Menlo, "SF Mono", Consolas, monospace; font-size: 12.5px; word-break: break-all; text-align: left; }
.figure { background: var(--panel); border: 1px solid var(--rule); border-radius: 3px; padding: 10px 12px 4px; margin: 14px 0 8px; }
.figure figcaption { font-size: 13px; color: var(--ink-soft); padding: 2px 2px 8px; }
.chart { width: 100%; height: auto; display: block; }
.grid { stroke: var(--rule); stroke-width: 1; }
.axis { stroke: var(--ink); stroke-width: 1; }
.tick { font-size: 10px; fill: var(--ink-soft); font-family: Menlo, "SF Mono", Consolas, monospace; }
.axis-label { font-size: 11px; fill: var(--ink-soft); }
.residue { font-size: 10px; font-family: Menlo, "SF Mono", Consolas, monospace; }
.row-label { font-size: 12px; fill: var(--ink); }
.row-value { font-size: 11px; fill: var(--ink-soft); font-family: Menlo, "SF Mono", Consolas, monospace; }
.dens-fill { fill-opacity: 0.07; }
.dens-line { fill: none; stroke-width: 1.6; }
.series-key { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 0; }
.series-key button {
  font: inherit; font-size: 13px; cursor: pointer; padding: 3px 11px;
  background: var(--panel); border: 1px solid var(--rule); border-radius: 999px; color: var(--ink);
  display: inline-flex; align-items: center; gap: 7px;
}
.series-key button::before {
  content: ""; width: 9px; height: 9px; border-radius: 50%; background: var(--dot);
}
.series-key button[aria-pressed="false"] { color: #a4aca8; border-style: dashed; }
.series-key button[aria-pressed="false"]::before { background: #c8cec9; }
.flag { border-left: 3px solid #d6403a; padding: 2px 0 2px 12px; margin: 10px 0; }
.ok { border-left: 3px solid #2e9e5b; padding: 2px 0 2px 12px; margin: 10px 0; }
.grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
ul.tight { margin: 6px 0 14px; padding-left: 20px; }
ul.tight li { margin: 2px 0; }
code { font-family: Menlo, "SF Mono", Consolas, monospace; font-size: 13px; background: #eaeee9; padding: 1px 4px; border-radius: 2px; }
footer { margin-top: 64px; padding-top: 16px; border-top: 1px solid var(--rule); color: var(--ink-soft); font-size: 13px; }
@media (max-width: 620px) {
  .band-name { min-width: 0; }
  h1 { font-size: 24px; }
}
"""

JS = """
document.querySelectorAll('.series-key button').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var name = btn.dataset.series;
    var on = btn.getAttribute('aria-pressed') !== 'true';
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    document.querySelectorAll('[data-series="' + CSS.escape(name) + '"]').forEach(function (el) {
      el.style.display = on ? '' : 'none';
    });
  });
});
"""


def residue_legend():
    items = "".join(
        '<span style="color:%s">%s</span>' % (CLASS_COLOUR[key], esc(label))
        for key, label in CLASS_LABEL.items()
    )
    return '<div class="legend">%s</div>' % items


def series_key(datasets):
    buttons = "".join(
        '<button type="button" aria-pressed="true" data-series="%s" style="--dot:%s">%s</button>'
        % (esc(ds.label), SERIES_COLOURS[i % len(SERIES_COLOURS)], esc(ds.label))
        for i, ds in enumerate(datasets)
    )
    return '<div class="series-key">%s</div>' % buttons


def table(headers, rows, numeric_from=1):
    head = "".join(
        "<th%s>%s</th>" % (' class="num"' if i >= numeric_from else "", esc(h))
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            "<td%s>%s</td>" % (' class="num"' if i >= numeric_from else "", cell)
            for i, cell in enumerate(row)
        )
        body.append("<tr>%s</tr>" % cells)
    return "<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>" % (head, "".join(body))


def figure(svg, caption):
    return '<figure class="figure">%s<figcaption>%s</figcaption></figure>' % (svg, esc(caption))


def build_report(datasets, positive, negative, ph, kmer_k, top_kmers):
    out = []
    add = out.append

    generated = datetime.now().strftime("%d %B %Y, %H:%M")
    total_seqs = sum(len(ds.sequences) for ds in datasets)

    add("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    add("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    add("<title>Peptide dataset EDA</title>")
    add("<style>%s</style></head><body><div class='wrap'>" % CSS)

    # ---- Hero -------------------------------------------------------------
    add("<h1>Peptide dataset EDA</h1>")
    add("<p class='meta'>%s sequences across %d files, generated %s. Charge and pI at pH %.1f.</p>"
        % (fmt(total_seqs), len(datasets), esc(generated), ph))
    add("<div class='hero'>%s</div>" % sequence_band(datasets))
    add(residue_legend())
    add("<p class='note'>One representative sequence per file, coloured by residue class. "
        "The same colours identify residues in every figure below.</p>")

    # ---- Section: what is in each file -----------------------------------
    add("<h2>What is in each file</h2>")
    rows = []
    for ds in datasets:
        comp = ds.composition()
        rows.append([
            esc(ds.label),
            fmt(ds.n_records),
            fmt(len(ds.unique_sequences)),
            fmt(ds.n_duplicate_rows),
            fmt(ds.summary["length"]["median"], 0),
            "%s - %s" % (fmt(ds.summary["length"]["min"], 0), fmt(ds.summary["length"]["max"], 0)),
            fmt(ds.total_residues),
            fmt(ds.file_size / 1024, 0) + " KB",
        ])
    add(table(
        ["File", "Records", "Unique seqs", "Duplicate rows", "Median length",
         "Length range", "Residues", "Size"],
        rows,
    ))
    paths = "".join("<li><code>%s</code> read as <b>%s</b></li>" % (esc(ds.path), esc(ds.label))
                    for ds in datasets)
    add("<ul class='tight'>%s</ul>" % paths)

    add("<h3>Header format</h3>")
    header_rows = []
    for ds in datasets:
        for h in ds.sample_headers:
            header_rows.append([esc(ds.label), "<span class='seq'>%s</span>" % esc(h)])
    add(table(["File", "First headers"], header_rows, numeric_from=99))

    # ---- Section: data quality -------------------------------------------
    add("<h2>Data quality</h2>")
    add("<p>Anything flagged here changes what you can train on. Duplicates inflate "
        "apparent dataset size, non-standard residues break one-hot encoders, and "
        "sequences shared between a positive and a negative file are label noise.</p>")

    quality_rows = []
    for ds in datasets:
        nonstd = ", ".join("%s x%d" % (a, c) for a, c in sorted(ds.nonstandard.items())) or "none"
        quality_rows.append([
            esc(ds.label),
            fmt(ds.n_empty),
            fmt(ds.n_duplicate_headers),
            fmt(ds.n_duplicate_rows),
            fmt(ds.n_seq_with_nonstandard),
            nonstd,
            fmt(len(ds.warnings)),
        ])
    add(table(
        ["File", "Empty", "Repeated headers", "Repeated seqs", "Seqs with non-standard",
         "Non-standard residues", "Parse warnings"],
        quality_rows,
    ))

    flags = []
    for ds in datasets:
        if ds.n_duplicate_rows:
            top = ", ".join("%s x%d" % (s[:18] + ("..." if len(s) > 18 else ""), c)
                            for s, c in ds.top_duplicates if c > 1)
            flags.append("%s has %s whose sequence already appeared. Most repeated: %s."
                         % (ds.label, plural(ds.n_duplicate_rows, "record"), top))
        if ds.nonstandard:
            flags.append("%s uses non-standard codes: %s. Decide whether to drop, map or keep them."
                         % (ds.label, ", ".join(sorted(ds.nonstandard))))
        if ds.n_empty:
            flags.append("%s has %s with no sequence." % (ds.label, plural(ds.n_empty, "header")))
        short = sum(1 for v in ds.features["length"] if v < 5)
        if short:
            flags.append("%s has %s shorter than 5 residues."
                         % (ds.label, plural(short, "sequence")))
    if flags:
        add("".join("<div class='flag'>%s</div>" % esc(f) for f in flags))
    else:
        add("<div class='ok'>No duplicates, empty records or non-standard residues found.</div>")

    warn_rows = []
    for ds in datasets:
        for w in ds.warnings[:6]:
            warn_rows.append([esc(ds.label), esc(w)])
    if warn_rows:
        add("<h3>Parser warnings</h3>")
        add(table(["File", "Warning"], warn_rows, numeric_from=99))

    # ---- Section: overlap -------------------------------------------------
    if len(datasets) > 1:
        add("<h2>Overlap between files</h2>")
        add("<p>Each cell is the share of the row file's unique sequences that also appear "
            "in the column file. A high value on a positive-negative pair means contradictory "
            "labels. A high value against a held-out set means leakage.</p>")
        head = ["Share of row found in"] + [ds.label for ds in datasets]
        rows = []
        for a in datasets:
            cells = [esc(a.label)]
            for b in datasets:
                if a is b:
                    cells.append("<span class='num'>-</span>")
                    continue
                shared = len(a.unique_sequences & b.unique_sequences)
                pct = shared / len(a.unique_sequences) * 100 if a.unique_sequences else 0.0
                cells.append("%.1f%% <span class='tick'>(%s)</span>" % (pct, fmt(shared)))
            rows.append(cells)
        add(table(head, rows))

        overlaps = []
        for i, a in enumerate(datasets):
            for b in datasets[i + 1:]:
                shared = len(a.unique_sequences & b.unique_sequences)
                if shared:
                    overlaps.append("%s and %s share %s exactly."
                                    % (a.label, b.label, plural(shared, "sequence")))
        if overlaps:
            add("".join("<div class='flag'>%s</div>" % esc(o) for o in overlaps))
        else:
            add("<div class='ok'>No exact sequence is shared between any two files.</div>")

    # ---- Section: length --------------------------------------------------
    add("<h2>Length</h2>")
    add("<p>Length drives padding, truncation and the maximum context your generator "
        "needs to produce. Check the upper tail before fixing a maximum length.</p>")
    add(figure(
        svg_distribution({ds.label: ds.features["length"] for ds in datasets},
                         "Length", "residues"),
        "Length distribution, normalised to the share of each file's sequences per bin.",
    ))
    add(series_key(datasets))
    rows = []
    for ds in datasets:
        s = ds.summary["length"]
        rows.append([esc(ds.label)] + [fmt(s[k], 0) for k in
                    ("min", "p05", "p25", "median", "p75", "p95", "max")] +
                    [fmt(s["mean"], 1), fmt(s["sd"], 1)])
    add(table(["File", "Min", "5%", "25%", "Median", "75%", "95%", "Max", "Mean", "SD"], rows))

    # ---- Section: composition --------------------------------------------
    add("<h2>Residue composition</h2>")
    add("<p>Bars are ordered by residue class rather than alphabetically, so a shift "
        "between files shows up as a block moving rather than scattered single bars.</p>")
    for ds in datasets:
        add(figure(svg_composition(ds.composition(), ds.label), "%s residue frequencies." % ds.label))

    if positive and negative:
        pc = positive.composition()
        nc = negative.composition()
        eps = 1e-6
        enrich = {a: math.log2((pc.get(a, 0.0) + eps) / (nc.get(a, 0.0) + eps)) for a in STANDARD_AA}
        add("<h3>Enrichment in %s relative to %s</h3>" % (esc(positive.label), esc(negative.label)))
        add(figure(svg_enrichment(enrich), "Positive log2 values mean the residue is more "
                                           "common in %s." % positive.label))
        gains = sorted(enrich.items(), key=lambda t: -t[1])
        add("<p>Most enriched: %s. Most depleted: %s.</p>" % (
            esc(", ".join("%s (%+.2f)" % (a, v) for a, v in gains[:5])),
            esc(", ".join("%s (%+.2f)" % (a, v) for a, v in gains[-5:][::-1])),
        ))

    # ---- Section: terminals ----------------------------------------------
    add("<h2>Terminal residues</h2>")
    add("<p>Many curated peptide sets carry an artefact at one terminus, for example a "
        "methionine start left over from translation or a shared cloning tag.</p>")
    rows = []
    for ds in datasets:
        n_top = ", ".join("%s %.0f%%" % (a, c / max(len(ds.sequences), 1) * 100)
                          for a, c in ds.nterm.most_common(4))
        c_top = ", ".join("%s %.0f%%" % (a, c / max(len(ds.sequences), 1) * 100)
                          for a, c in ds.cterm.most_common(4))
        rows.append([esc(ds.label), esc(n_top), esc(c_top)])
    add(table(["File", "Most common first residue", "Most common last residue"], rows, numeric_from=99))

    # ---- Section: physicochemistry ---------------------------------------
    add("<h2>Physicochemical profile</h2>")
    add("<p>These are the properties the antimicrobial literature keeps returning to: "
        "cationic charge, hydrophobicity, and the amphipathicity that lets a helix sit "
        "in a membrane with its polar face out.</p>")

    for key, label, unit in FEATURE_ORDER:
        if key in ("length", "max_run", "max_residue_frac"):
            continue
        add("<h3>%s</h3>" % esc(label))
        add(figure(
            svg_distribution({ds.label: ds.features[key] for ds in datasets},
                             label, unit.format(ph=ph)),
            "%s across files." % label,
        ))
    add(series_key(datasets))

    add("<h3>Summary of every feature</h3>")
    for ds in datasets:
        rows = []
        for key, label, unit in FEATURE_ORDER:
            s = ds.summary[key]
            digits = 0 if key in ("length", "molecular_weight", "max_run") else 2
            rows.append([esc(label)] + [fmt(s[k], digits) for k in
                        ("mean", "sd", "min", "p25", "median", "p75", "max")])
        add("<h3>%s</h3>" % esc(ds.label))
        add(table(["Feature", "Mean", "SD", "Min", "25%", "Median", "75%", "Max"], rows))

    # ---- Section: separability -------------------------------------------
    if positive and negative:
        add("<h2>How separable are %s and %s</h2>" % (esc(positive.label), esc(negative.label)))
        add("<p>Each bar is the AUROC of a single feature used alone as a classifier, "
            "measured from the 0.5 no-information line. A feature near 0.5 carries nothing "
            "on its own. A feature near 0 or 1 is a strong signal, and if it is something "
            "trivial like length, it is usually a sampling artefact in how the negatives "
            "were built rather than biology.</p>")
        scored = []
        for key, label, unit in FEATURE_ORDER:
            a = positive.features[key]
            b = negative.features[key]
            scored.append((key, label, auroc(a, b), cohens_d(a, b)))
        scored.sort(key=lambda t: -abs(t[2] - 0.5))
        add(figure(
            svg_separability([(label, value, d) for _, label, value, d in scored]),
            "Single-feature AUROC, %s as the positive class." % positive.label,
        ))
        add(table(
            ["Feature", "AUROC", "Cohen d", "Mean " + positive.label, "Mean " + negative.label],
            [[esc(label),
              fmt(value, 3),
              fmt(d, 2),
              fmt(mean(positive.features[key]), 2),
              fmt(mean(negative.features[key]), 2)]
             for key, label, value, d in scored],
        ))

    # ---- Section: k-mers --------------------------------------------------
    add("<h2>Recurring %d-mers</h2>" % kmer_k)
    add("<p>Short repeated motifs point at either real structural units or near-duplicate "
        "families that will make a random train and test split look better than it is.</p>")
    for ds in datasets:
        top = ds.kmers.most_common(top_kmers)
        rows = [[
            "<span class='seq'>%s</span>" % esc(k),
            fmt(c),
            fmt(c / max(ds.kmer_total, 1) * 1000, 2),
        ] for k, c in top]
        add("<h3>%s</h3>" % esc(ds.label))
        add(table(["%d-mer" % kmer_k, "Count", "Per 1000 %d-mers" % kmer_k], rows))

    if positive and negative:
        eps = 1.0
        pos_total = max(positive.kmer_total, 1)
        neg_total = max(negative.kmer_total, 1)
        scores = []
        for kmer, count in positive.kmers.items():
            if count < 5:
                continue
            pf = count / pos_total
            nf = (negative.kmers.get(kmer, 0) + eps) / neg_total
            scores.append((kmer, math.log2(pf / nf), count))
        scores.sort(key=lambda t: -t[1])
        add("<h3>%d-mers most over-represented in %s</h3>" % (kmer_k, esc(positive.label)))
        add(table(
            ["%d-mer" % kmer_k, "log2 enrichment", "Count in %s" % positive.label,
             "Count in %s" % negative.label],
            [["<span class='seq'>%s</span>" % esc(k), fmt(v, 2), fmt(c),
              fmt(negative.kmers.get(k, 0))]
             for k, v, c in scores[:top_kmers]],
        ))

    # ---- Section: extremes ------------------------------------------------
    add("<h2>Edge cases worth eyeballing</h2>")
    for ds in datasets:
        rows = [[esc(kind), "<span class='seq'>%s</span>" % esc(seq[:90] + ("..." if len(seq) > 90 else "")), esc(note)]
                for kind, seq, note in ds.extremes()]
        if rows:
            add("<h3>%s</h3>" % esc(ds.label))
            add(table(["Kind", "Sequence", "Value"], rows, numeric_from=99))

    low_complexity = []
    for ds in datasets:
        n = sum(1 for v in ds.features["max_residue_frac"] if v > 0.5)
        runs = sum(1 for v in ds.features["max_run"] if v >= 6)
        low_complexity.append([esc(ds.label), fmt(n), fmt(runs),
                               fmt(ds.summary["entropy"]["median"], 2)])
    add("<h3>Low complexity</h3>")
    add(table(["File", "Seqs dominated by one residue (>50%)", "Seqs with a run of 6 or more",
               "Median entropy (bits)"], low_complexity))

    # ---- Footer -----------------------------------------------------------
    add("<footer>")
    add("<p>Charge and isoelectric point use the EMBOSS pKa set. GRAVY uses Kyte-Doolittle. "
        "The hydrophobic moment is the Eisenberg consensus scale over sliding windows of 11 "
        "residues at 100 degrees per residue. AUROC is rank based with tie correction. "
        "Non-standard residues contribute zero to scale-based features and are excluded from "
        "mass, so files carrying many of them will read slightly light.</p>")
    add("<p>Regenerate with <code>python amp_eda.py %s</code></p>"
        % esc(" ".join("%s=%s" % (ds.label, ds.path) for ds in datasets)))
    add("</footer>")
    add("</div><script>%s</script></body></html>" % JS)
    return "".join(out)


# ---------------------------------------------------------------------------
# Section 8. Command line
# ---------------------------------------------------------------------------

POSITIVE_HINTS = ("antibact", "amp", "positive", "pos", "active")
NEGATIVE_HINTS = ("background", "negative", "neg", "random", "decoy")


def resolve_inputs(specs):
    resolved = []
    for spec in specs:
        if "=" in spec and not os.path.exists(spec):
            label, path = spec.split("=", 1)
        else:
            label = os.path.splitext(os.path.basename(spec))[0]
            path = spec
        if not os.path.exists(path):
            raise SystemExit("File not found: %s" % path)
        resolved.append((label, path))
    return resolved


def pick_class(datasets, explicit, hints):
    if explicit:
        for ds in datasets:
            if ds.label == explicit:
                return ds
        raise SystemExit("No dataset labelled %r. Available: %s"
                         % (explicit, ", ".join(ds.label for ds in datasets)))
    for hint in hints:
        for ds in datasets:
            if hint in ds.label.lower():
                return ds
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exploratory data analysis for peptide FASTA files.")
    parser.add_argument("fasta", nargs="+",
                        help="FASTA paths, or label=path to name a dataset explicitly")
    parser.add_argument("--out", default="amp_eda_report.html",
                        help="output HTML path (default: amp_eda_report.html)")
    parser.add_argument("--ph", type=float, default=7.0,
                        help="pH for net charge (default: 7.0)")
    parser.add_argument("--kmer", type=int, default=3, help="k-mer size (default: 3)")
    parser.add_argument("--top-kmers", type=int, default=15,
                        help="how many k-mers to list per table (default: 15)")
    parser.add_argument("--positive", default=None,
                        help="label of the positive class for the separability section")
    parser.add_argument("--negative", default=None,
                        help="label of the negative class for the separability section")
    parser.add_argument("--json", default=None,
                        help="also write the computed summary statistics to this JSON path")
    args = parser.parse_args(argv)

    datasets = []
    for label, path in resolve_inputs(args.fasta):
        print("reading %s from %s" % (label, path), file=sys.stderr)
        ds = Dataset(label, path, args.ph, args.kmer)
        print("  %d records, %d unique sequences, %d residues"
              % (ds.n_records, len(ds.unique_sequences), ds.total_residues), file=sys.stderr)
        datasets.append(ds)

    positive = pick_class(datasets, args.positive, POSITIVE_HINTS)
    negative = pick_class(datasets, args.negative, NEGATIVE_HINTS)
    if positive is negative:
        negative = None
    if positive and negative:
        print("comparing %s (positive) against %s (negative)"
              % (positive.label, negative.label), file=sys.stderr)

    report = build_report(datasets, positive, negative, args.ph, args.kmer, args.top_kmers)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report)
    print("wrote %s (%.0f KB)" % (args.out, os.path.getsize(args.out) / 1024), file=sys.stderr)

    if args.json:
        payload = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "ph": args.ph,
            "datasets": {},
        }
        for ds in datasets:
            payload["datasets"][ds.label] = {
                "path": ds.path,
                "records": ds.n_records,
                "unique_sequences": len(ds.unique_sequences),
                "duplicate_rows": ds.n_duplicate_rows,
                "total_residues": ds.total_residues,
                "nonstandard_residues": ds.nonstandard,
                "composition": ds.composition(),
                "features": ds.summary,
            }
        if positive and negative:
            payload["separability"] = {
                key: {
                    "auroc": auroc(positive.features[key], negative.features[key]),
                    "cohens_d": cohens_d(positive.features[key], negative.features[key]),
                }
                for key, _, _ in FEATURE_ORDER
            }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("wrote %s" % args.json, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())