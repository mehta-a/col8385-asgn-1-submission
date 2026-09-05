"""Entry point for `uv run train`.

Trains the conditional VAE on training.fasta and writes two files:

    checkpoint/model.pt     the weights, the property standardiser, and the
                            empirical length and property table used at
                            sampling time
    checkpoint/model.json   a metadata stub, so anything looking for the
                            original JSON checkpoint path still finds a file

Every option of peptide_vae's train subcommand is accepted here, so
`uv run train --epochs 200 --latent 48` works as expected.
"""

import sys

from .peptide_vae import main as _vae_main


def main() -> int:
    return _vae_main(["train"] + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())