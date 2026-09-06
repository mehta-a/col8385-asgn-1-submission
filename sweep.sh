#!/usr/bin/env bash
# Generate and score every training snapshot, then print one comparison table.
#
# Run from the submission repo root, with amp-grader-demo as a sibling directory.
# Scores each snapshot by calling the scorer directly, so nothing needs to be
# committed or pushed until you have picked a winner.
#
#   bash sweep.sh runs/v1
#
set -euo pipefail

RUN_DIR="${1:-runs/v1}"
GRADER="${GRADER:-../amp-grader-demo}"
SUBMISSION="$(pwd)"

if [ ! -d "$GRADER" ]; then
    echo "grader not found at $GRADER. Set GRADER=/path/to/amp-grader-demo" >&2
    exit 1
fi

shopt -s nullglob
SNAPSHOTS=("$RUN_DIR"/snapshot-e*.pt)
if [ ${#SNAPSHOTS[@]} -eq 0 ]; then
    echo "no snapshots in $RUN_DIR. Train with --snapshot-every 25" >&2
    exit 1
fi

mkdir -p sweep

for SNAP in "${SNAPSHOTS[@]}"; do
    TAG="$(basename "$SNAP" .pt)"
    LIB="$SUBMISSION/sweep/$TAG.fasta"
    METRICS="$SUBMISSION/sweep/$TAG.metrics.json"

    echo
    echo "=== $TAG ==============================================="

    uv run generate --model "$SNAP" --out "$LIB" --n-sequences 1000

    echo "--- local feature match"
    uv run evaluate "$SNAP" "$LIB" | tail -n 12

    echo "--- scorer"
    ( cd "$GRADER" && uv run --project . python scorer/scorer.py \
        --library-fasta "$LIB" \
        --profile grader \
        --training-fasta data/training/training.fasta \
        --reference-amps-fasta data/training/training.fasta \
        --reference-generic-fasta data/generic/background.fasta \
        --out "$METRICS" ) || echo "scoring failed for $TAG"
done

echo
echo "=== summary ============================================="
uv run python - "$SUBMISSION/sweep" <<'PY'
import json, pathlib, sys

rows = []
for path in sorted(pathlib.Path(sys.argv[1]).glob("*.metrics.json")):
    blob = json.loads(path.read_text())
    rows.append((path.stem.replace(".metrics", ""), blob))

if not rows:
    print("no metrics files found")
    raise SystemExit(0)


def dig(blob, name):
    """The scorer nests metrics differently between versions, so search."""
    stack = [blob]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            label = node.get("name") or node.get("metric")
            if isinstance(label, str) and label.lower().startswith(name.lower()):
                if isinstance(node.get("value"), (int, float)):
                    return node["value"]
            for key, value in node.items():
                if key.lower().startswith(name.lower()):
                    if isinstance(value, (int, float)):
                        return value
                    if isinstance(value, dict) and isinstance(value.get("value"), (int, float)):
                        return value["value"]
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return None


WANTED = ["Diversity", "Novelty", "FBD (AMPs)", "FBD (generic)",
          "Activity", "Authenticity", "Conformity"]

header = "%-16s" % "snapshot" + "".join("%16s" % w[:15] for w in WANTED)
print(header)
print("-" * len(header))
for tag, blob in rows:
    line = "%-16s" % tag
    for want in WANTED:
        value = dig(blob, want)
        line += "%16s" % ("%.4f" % value if isinstance(value, (int, float)) else "-")
    print(line)
print()
print("Lower is better for both FBD columns. Higher is better for the rest.")
PY