# amp-model

A conditional variational autoencoder for antimicrobial peptide generation.

`uv run train` fits the model on `training.fasta`; `uv run generate` writes 1,000
novel candidates to `generate/library.fasta`. Both are deterministic: the same seed
produces byte-identical output across processes and machines.

```
uv run train --epochs 120 --out runs/v1
uv run generate --model runs/v1/model.pt --exclude-fasta data/training/training.fasta
```

---

## 1. What the model is replacing

The starter kit ships a unigram sampler: draw a length from the empirical length
distribution, then draw that many residues independently from marginal amino acid
frequencies. It matches composition and length exactly, and destroys everything else.

That matters because the signal in this dataset is ordering, not composition. The
enrichment analysis on the training set put motifs like `LKK`, `KKL`, `KLL`, `RRR`,
`PRP` and the tryptophan patterns at log2 enrichments above +5 against background.
None of that survives independent sampling. The unigram model is the right floor to
measure against, and it is the thing any structure-aware metric should separate us from.

So the requirement is a model that captures *which residues co-occur and in what
arrangement*, while still reproducing the marginals the unigram gets for free.

---

## 2. Why a VAE

Three properties made it the right shape for this problem.

**It gives a continuous latent space, not just a sampler.** Every peptide maps to a
point in a 32-dimensional space, and nearby points decode to related peptides. That
means generation is interpolation inside a learned manifold rather than sampling from
a table, which is what produces sequences that are novel but plausible instead of
novel but random.

**Its training objective is explicitly distribution matching.** The ELBO is a bound on
the data log-likelihood. We are graded on how closely the generated library resembles
the real distribution, so optimising a likelihood bound is optimising close to the
thing being measured. A GAN would optimise a discriminator's opinion instead, and mode
collapse is exactly the failure the diversity metrics punish.

**Sampling is one forward pass.** We need 1,000 accepted sequences out of a rejection
loop, under a wall-clock cap, on CPU. An autoregressive language model needs L
sequential steps per sequence; a diffusion model needs tens of denoising passes. The
VAE decodes an entire batch in a single pass, which leaves budget for rejection
sampling and refinement.

The cost is the well-known one: VAE samples are blurrier than autoregressive samples.
Sections 4 and 6 are mostly about buying that sharpness back.

---

## 3. Architecture

```
peptide (KWKLFKKIEKVGQNIR)
        |
   tokenise + pad
        |
   Encoder: 4 gated conv blocks  <--- conditioning: length + 4 properties
        |
   mu (32), logvar (32)
        |
   z = mu + sigma * eps          <--- KL term attaches here
        |
   Decoder: positionwise         <--- same conditioning
        |
   logits over the 20-letter alphabet, per position
        |
   peptide                       <--- reconstruction term attaches here
```

**Encoder.** Four gated convolutional blocks over the embedded sequence, pooled to a
fixed-width vector, projected to `mu` and `logvar`. Convolutions rather than a GRU
because the signal we care about is local motif structure (3-5 residue windows), which
is what a stack of dilated convolutions represents directly, and because they
parallelise across positions instead of running a recurrence per position.

**Latent.** 32 dimensions by default (`--latent`). Large enough to carry motif identity
and amphipathic arrangement; small enough that the KL term is a real constraint rather
than a rounding error.

**Conditioning.** Length, net charge, hydrophobic moment, GRAVY, and cysteine fraction.
All five are computed from the sequences themselves by `amp_eda.compute_features`, so
there is no external label file and nothing to leak. Properties are standardised, and
the standardiser is saved into the checkpoint so training and sampling agree.

**Decoder.** Positionwise. Takes `z` plus the conditioning vector and emits independent
logits for every position at once. It is deliberately *not* autoregressive, which is
the one design choice here worth arguing about.

---

## 4. The decoder choice, and posterior collapse

The classic failure of sequence VAEs is posterior collapse. If the decoder is a strong
autoregressive model, it can reconstruct the sequence perfectly well from its own
previous outputs, without ever reading `z`. Gradient descent notices this: the KL term
is a cost with no offsetting benefit, so KL is driven to zero, the encoder becomes
uninformative, and sampling the prior gives you the dataset average forever. You end up
with an expensive unigram model.

A positionwise decoder cannot do this. There is no path from the input sequence to the
output except through `z`, so `z` has to carry the information or reconstruction fails.
Collapse becomes structurally difficult rather than something we fight with tricks.

The price is that residues are conditionally independent given `z`. The model gets the
global picture right (this is a cationic amphipathic helix of length 18) but can emit
locally incoherent combinations, because nothing couples position 7 to position 8 except
their shared latent.

Three mitigations, in order of how much they matter:

**Refinement at sampling time (`--refine`, default 1).** Decode from `z`, re-encode the
result, decode again from the posterior mean. The first pass places the sequence in the
right region; the second pass has an actual sequence to condition on and cleans up the
incoherent combinations. One round is usually enough. This is cheap iterative refinement
standing in for the coupling the decoder lacks.

**Free bits (`--free-bits`, 0.1 nats per dimension).** The KL penalty is floored per
dimension, so a dimension that is currently carrying less than 0.1 nats is not penalised
for existing. Without this, dimensions that are merely slow to become useful get switched
off early and never come back.

**Cyclical KL annealing, 4 cycles.** Rather than one monotonic ramp of beta from 0 to 1,
the schedule cycles. Each cycle starts with a near-autoencoder phase where the model is
free to learn a rich encoding, then tightens it. Monotonic annealing gives one shot at
finding a good encoding; cycling gives four, and the later cycles start from a better
initialisation than the first.

---

## 5. Length as a condition, not a prediction

Length is fed in, not generated. The decoder is told how long the peptide is and fills
in that many positions.

This is a deliberate trade. A model that predicts its own length has one more thing to
get wrong, and length distribution mismatch is directly visible to the scorer. By
conditioning instead, and drawing lengths from the empirical training distribution at
sampling time, the generated length distribution matches training by construction. We
spend the model's capacity on the hard part (which residues, in what order) and take the
easy part for free.

The same logic applies to the property conditioning. At sampling time, `(length,
properties)` tuples are drawn *jointly from real training rows*, not independently per
field. Charge and length and hydrophobicity are correlated in real peptides; sampling
them independently would ask the decoder for combinations that never occur in nature,
and it would oblige with something off-manifold.

---

## 6. The prior hole problem, and the fitted prior

The KL term pulls each posterior toward N(0, I), but the *aggregate* posterior -- the
distribution you get by encoding the whole training set -- is never exactly N(0, I).
It has a shifted mean, correlated dimensions, and holes: regions with high prior density
that no training peptide ever maps to. Sample there and the decoder produces something
it was never trained to produce.

After training we fit a full-covariance Gaussian to the aggregate posterior and sample
from that instead (`--prior fitted`, the default; `--prior normal` for the textbook
version). It costs one forward pass over the training set at the end of training and it
measurably reduces off-manifold samples. `--temperature` (default 0.9) shrinks the
sampling distribution slightly for the same reason: trading a little diversity for
staying inside the region the decoder understands.

---

## 7. Data handling

`peptide_data.py` does three things worth knowing about.

**Cluster-aware splitting.** Peptide datasets are full of near-duplicates -- variants of
the same natural peptide, truncations, single-residue mutants. A random train/validation
split puts near-identical sequences on both sides, and validation accuracy then measures
memorisation rather than generalisation. Sequences are clustered by MinHash over k-mers
and the split is made at cluster level, so the validation number means something.

**Stable hashing.** The MinHash uses CRC32, not Python's built-in `hash()`. The built-in
is salted per process, which would make the clustering, and therefore the split, differ
between runs. That would have broken the byte-identical rerun requirement in a way that
only shows up intermittently.

**Run filtering on input.** Homopolymer runs above `--max-run` are dropped at load time.
The training set contains things like long poly-K stretches; they score wonderfully on
net charge and terribly on everything else, and the model will happily learn to produce
them.

---

## 8. Generation loop

`generate.py` keeps the starter kit's round-based structure: sample a batch, keep what
passes, advance the seed, repeat until 1,000 are collected. Each round's seed is derived
from `--seed`, so the whole loop is deterministic.

The original loop was a rejection sampler with nothing to reject on except duplicates.
The accept predicate now also rejects:

- lengths outside [8, 50]
- homopolymer runs longer than `--max-run` (default 5)
- sequences more than `--max-residue-frac` of a single residue (default 0.5)
- optionally, a crude net charge below `--min-charge` (off by default, since forcing it
  distorts the charge distribution the scorer is comparing against)

The exclusion set defaults to `training.fasta` rather than `antibacterial.fasta`. The
antibacterial set is entirely contained in training, so excluding against training is a
strict superset and needs one fewer input file.

---

## 9. Checkpoints

Training writes two files:

| File | Contents |
| --- | --- |
| `checkpoint/model.pt` | weights, property standardiser, fitted prior, and the empirical (length, property) table used at sampling time |
| `checkpoint/model.json` | metadata stub, so anything hard-coded to the starter kit's JSON path still finds a file |

The stub exists because the harness expects a JSON checkpoint and the VAE needs binary
weights. `generate.py` resolves which one it has been handed and falls back to the
empirical sampler if only the JSON is present, so a bare `uv run generate` never crashes.

---

## 10. Reading the training log

`active` is the count of latent dimensions carrying more than the free-bits floor.

- `active` at 0: the posterior has collapsed. Raise `--free-bits` or lower `--beta`.
- `active` at 32/32 with validation accuracy above 0.95: the KL term is doing nothing and
  this is close to a plain autoencoder. Prior samples will be off-manifold. Raise `--beta`.
- Somewhere in between, with validation accuracy in the high 0.8s: healthy.

For output quality, `uv run evaluate runs/v1/model.pt generate/library.fasta` reports
per-feature AUROC of generated against training. A classifier that cannot tell them apart
scores 0.5 on every feature, which is the target. Anything above about 0.65 names the
axis that needs fixing.

---

## 11. Known limitations

- Residues are conditionally independent given `z`. `--refine` mitigates this; it does not
  eliminate it. An autoregressive or diffusion decoder would produce locally sharper
  sequences at the cost of collapse risk and sampling time.
- The conditioning properties are hand-chosen physicochemical descriptors. They cover the
  axes the scorer looks at, but nothing about structure or target specificity.
- Nothing in the objective optimises for activity. The model matches the training
  distribution; it does not push toward better peptides than the ones it was shown. That
  would need a reward-driven method (a GFlowNet, or latent-space optimisation against a
  learned activity proxy) on top of this.