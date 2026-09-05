"""Entry point for `uv run evaluate`, and for ad hoc sampling.

`generate` is the submission path and writes generate/library.fasta.
This module exposes the diagnostic commands instead.
"""

import sys

from .peptide_vae import main as _vae_main


def sample() -> int:
    return _vae_main(["sample"] + sys.argv[1:])


def evaluate() -> int:
    return _vae_main(["evaluate"] + sys.argv[1:])