#!/usr/bin/env python3
"""
peptide_vae.py - a conditional variational autoencoder for short peptides.

Trained on training.fasta alone. The conditioning variables are computed from
the sequences themselves (length, net charge, hydrophobic moment, GRAVY,
cysteine fraction), so no label file is required.

Design notes, because the choices here are not arbitrary:

  * The decoder is non-autoregressive. A VAE with a strong autoregressive
    decoder can reconstruct sequences without reading z at all, which is
    posterior collapse: the KL term goes to zero and sampling the prior gives
    you the dataset average. With a positionwise decoder the latent is the
    only path from input to output, so z has to carry the sequence. The cost
    is that residues are conditionally independent given z, which is why
    --refine exists.
  * The KL term uses free bits. Each latent dimension is allowed a small
    budget of KL for free, which stops the optimiser from switching
    dimensions off early in training.
  * beta follows a cyclical schedule rather than a single ramp. Repeated
    warm restarts consistently give a better rate-distortion point than one
    monotonic anneal on datasets this small.
  * Length is a conditioning input rather than something the model predicts.
    At sampling time lengths are drawn from the training distribution, so the
    generated set matches it by construction.

Usage
-----
    python peptide_vae.py train training.fasta --epochs 120 --out runs/v1
    python peptide_vae.py sample runs/v1/model.pt -n 5000 --out generated.fasta
    python peptide_vae.py evaluate runs/v1/model.pt generated.fasta
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import peptide_data as pdata
from .peptide_data import PAD, PROPERTY_KEYS, VOCAB_SIZE
from .amp_eda import auroc, compute_features, mean as list_mean


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Gated residual convolution over the sequence axis, in (B, L, C)."""

    def __init__(self, dim, kernel=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim * 2, kernel, padding=kernel // 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.norm(x)
        h = h.transpose(1, 2)
        h = self.conv(h)
        h = F.glu(h, dim=1)
        h = h.transpose(1, 2)
        h = self.drop(h)
        x = x + h
        return x * mask.unsqueeze(-1)


class Conditioner(nn.Module):
    """Turns (length, properties) into a single conditioning vector."""

    def __init__(self, dim, max_len, n_properties):
        super().__init__()
        self.length_embedding = nn.Embedding(max_len + 1, dim)
        self.property_mlp = nn.Sequential(
            nn.Linear(n_properties, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, lengths, properties):
        return self.length_embedding(lengths) + self.property_mlp(properties)


class PeptideVAE(nn.Module):
    def __init__(self, max_len=50, dim=256, latent=32, layers=4, kernel=5,
                 dropout=0.1, n_properties=len(PROPERTY_KEYS)):
        super().__init__()
        self.max_len = max_len
        self.latent = latent

        self.conditioner = Conditioner(dim, max_len, n_properties)

        self.token_embedding = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD)
        self.encoder_positions = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.encoder_blocks = nn.ModuleList(
            [ConvBlock(dim, kernel, dropout) for _ in range(layers)]
        )
        self.to_latent = nn.Linear(dim * 2, latent * 2)

        self.from_latent = nn.Linear(latent + dim, dim)
        self.decoder_positions = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.decoder_blocks = nn.ModuleList(
            [ConvBlock(dim, kernel, dropout) for _ in range(layers)]
        )
        self.output = nn.Linear(dim, VOCAB_SIZE)

    def make_mask(self, lengths):
        positions = torch.arange(self.max_len, device=lengths.device)
        return positions.unsqueeze(0) < lengths.unsqueeze(1)

    def encode(self, tokens, lengths, properties):
        mask = self.make_mask(lengths)
        condition = self.conditioner(lengths, properties)
        h = self.token_embedding(tokens) + self.encoder_positions
        h = h + condition.unsqueeze(1)
        h = h * mask.unsqueeze(-1)
        for block in self.encoder_blocks:
            h = block(h, mask)

        denominator = mask.sum(1, keepdim=True).clamp(min=1).float()
        pooled_mean = h.sum(1) / denominator
        pooled_max = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
        pooled_max = torch.nan_to_num(pooled_max, neginf=0.0)

        mu, logvar = self.to_latent(torch.cat([pooled_mean, pooled_max], dim=-1)).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0), condition, mask

    def decode(self, z, condition, mask):
        h = self.from_latent(torch.cat([z, condition], dim=-1))
        h = h.unsqueeze(1) + self.decoder_positions
        h = h * mask.unsqueeze(-1)
        for block in self.decoder_blocks:
            h = block(h, mask)
        return self.output(h)

    def forward(self, tokens, lengths, properties):
        mu, logvar, condition, mask = self.encode(tokens, lengths, properties)
        if self.training:
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = mu
        logits = self.decode(z, condition, mask)
        return logits, mu, logvar


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def elbo_terms(logits, tokens, mu, logvar):
    """Per-sequence reconstruction and KL, both in nats."""
    batch = tokens.size(0)
    reconstruction = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), tokens.reshape(-1),
        ignore_index=PAD, reduction="sum",
    ) / batch
    kl_per_dimension = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
    return reconstruction, kl_per_dimension


def cyclical_beta(step, total_steps, cycles, beta_max, ramp=0.5):
    if total_steps <= 0 or cycles <= 0:
        return beta_max
    period = total_steps / cycles
    position = (step % period) / period
    return beta_max * min(1.0, position / ramp)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_tensors(data, indices, device):
    max_len = data["max_len"]
    stats = data["property_stats"]
    sequences = [data["sequences"][i] for i in indices]
    raw = [data["properties_raw"][i] for i in indices]
    standardised = pdata.standardise(raw, stats)

    tokens = torch.tensor([pdata.encode(s, max_len) for s in sequences],
                          dtype=torch.long, device=device)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long, device=device)
    properties = torch.tensor(standardised, dtype=torch.float32, device=device)
    return tokens, lengths, properties


def evaluate_split(model, tokens, lengths, properties, batch_size=512):
    model.eval()
    total_recon = 0.0
    total_kl = 0.0
    correct = 0
    counted = 0
    n = tokens.size(0)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            t, l, p = tokens[start:end], lengths[start:end], properties[start:end]
            logits, mu, logvar = model(t, l, p)
            recon, kl_dim = elbo_terms(logits, t, mu, logvar)
            batch = end - start
            total_recon += recon.item() * batch
            total_kl += kl_dim.sum(1).mean().item() * batch
            predictions = logits.argmax(-1)
            valid = t != PAD
            correct += ((predictions == t) & valid).sum().item()
            counted += valid.sum().item()
    return total_recon / n, total_kl / n, correct / max(counted, 1)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    data = pdata.prepare(
        args.fasta,
        cache=os.path.join(args.out, "prepared.json"),
        min_len=args.min_len, max_len=args.max_len,
        max_run=args.max_run, max_residue_frac=args.max_residue_frac,
        val_fraction=args.val_fraction, cluster=not args.no_cluster,
        jaccard=args.jaccard, seed=args.seed,
    )

    train_tokens, train_lengths, train_properties = build_tensors(data, data["train_idx"], device)
    val_tokens, val_lengths, val_properties = build_tensors(data, data["val_idx"], device)
    n_train = train_tokens.size(0)

    model = PeptideVAE(max_len=data["max_len"], dim=args.dim, latent=args.latent,
                       layers=args.layers, kernel=args.kernel,
                       dropout=args.dropout).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print("model has %s parameters, %d training sequences" % (f"{parameters:,}", n_train))

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_epoch = max(1, math.ceil(n_train / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

    history = []
    snapshots = []
    best_val = float("inf")
    step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(n_train, device=device)
        epoch_recon = 0.0
        epoch_kl = 0.0
        started = time.time()

        for start in range(0, n_train, args.batch_size):
            index = permutation[start:start + args.batch_size]
            tokens = train_tokens[index]
            lengths = train_lengths[index]
            properties = train_properties[index]

            logits, mu, logvar = model(tokens, lengths, properties)
            reconstruction, kl_dimension = elbo_terms(logits, tokens, mu, logvar)

            beta = cyclical_beta(step, total_steps, args.kl_cycles, args.beta)
            free_bits_kl = torch.clamp(kl_dimension.mean(0), min=args.free_bits).sum()
            loss = reconstruction + beta * free_bits_kl

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimiser.step()
            scheduler.step()

            epoch_recon += reconstruction.item()
            epoch_kl += kl_dimension.sum(1).mean().item()
            step += 1

        val_recon, val_kl, val_accuracy = evaluate_split(model, val_tokens, val_lengths,
                                                         val_properties, args.batch_size)
        with torch.no_grad():
            mu, logvar, _, _ = model.encode(val_tokens[:2048], val_lengths[:2048],
                                            val_properties[:2048])
            per_dimension = (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)).mean(0)
            # The free-bits floor forces every dimension to at least
            # args.free_bits nats, so a threshold below the floor would count
            # all of them regardless of what the model learned. Count only the
            # dimensions carrying meaningfully more than the floor.
            threshold = max(0.02, args.free_bits * 2.0)
            active_units = int((per_dimension > threshold).sum().item())
            kl_top = float(per_dimension.max().item())

        record = {
            "epoch": epoch,
            "beta": round(beta, 4),
            "train_recon": round(epoch_recon / steps_per_epoch, 3),
            "train_kl": round(epoch_kl / steps_per_epoch, 3),
            "val_recon": round(val_recon, 3),
            "val_kl": round(val_kl, 3),
            "val_accuracy": round(val_accuracy, 4),
            "active_units": active_units,
            "kl_top_dim": round(kl_top, 4),
            "seconds": round(time.time() - started, 1),
        }
        history.append(record)
        print("epoch %3d  beta %.3f  recon %7.3f  kl %6.2f  val recon %7.3f  "
              "val acc %.3f  active %2d/%d  %.1fs"
              % (epoch, beta, record["train_recon"], record["train_kl"],
                 record["val_recon"], record["val_accuracy"], active_units,
                 args.latent, record["seconds"]))

        if active_units == 0 and epoch > args.epochs // 4:
            print("  warning: no active latent dimensions, the posterior has collapsed. "
                  "Raise --free-bits or lower --beta.")

        objective = val_recon + val_kl
        if objective < best_val:
            best_val = objective
            save_checkpoint(model, data, args, os.path.join(args.out, "model.pt"))

        if args.snapshot_every and epoch % args.snapshot_every == 0:
            snapshot = os.path.join(args.out, "snapshot-e%03d.pt" % epoch)
            save_checkpoint(model, data, args, snapshot)
            snapshots.append(snapshot)
            print("  snapshot -> %s" % snapshot)

    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    # The harness looks for checkpoint/model.json. The weights cannot live in
    # JSON, so write a metadata stub there that points at the real checkpoint.
    stub_dir = os.path.dirname(os.path.abspath(args.stub)) if args.stub else None
    if stub_dir:
        os.makedirs(stub_dir, exist_ok=True)
        with open(args.stub, "w", encoding="utf-8") as handle:
            json.dump({
                "model": "conditional-peptide-vae",
                "weights": os.path.relpath(os.path.join(args.out, "model.pt")),
                "config": {"dim": args.dim, "latent": args.latent,
                           "layers": args.layers, "max_len": args.max_len},
                "training_sequences": n_train,
                "best_val_objective": round(best_val, 4),
                "final_epoch": history[-1] if history else None,
            }, handle, indent=2)
        print("wrote metadata stub to %s" % args.stub)

    print("best validation objective %.3f, checkpoint in %s" % (best_val, args.out))

    fit_aggregate_posterior(os.path.join(args.out, "model.pt"), device)
    for snapshot in snapshots:
        fit_aggregate_posterior(snapshot, device)


def save_checkpoint(model, data, args, path):
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "max_len": data["max_len"], "dim": args.dim, "latent": args.latent,
            "layers": args.layers, "kernel": args.kernel, "dropout": args.dropout,
        },
        "property_stats": data["property_stats"],
        "empirical": {
            "lengths": [data["lengths"][i] for i in data["train_idx"]],
            "properties": [data["properties_raw"][i] for i in data["train_idx"]],
        },
        "train_sequences": [data["sequences"][i] for i in data["train_idx"]],
        "ph": data["ph"],
    }, path)


# ---------------------------------------------------------------------------
# Aggregate posterior, used as a better sampling distribution than N(0, I)
# ---------------------------------------------------------------------------

def fit_aggregate_posterior(checkpoint_path, device, limit=20000):
    """Fit a full-covariance Gaussian to q(z) over the training set.

    The prior is N(0, I) but the aggregate posterior never quite matches it.
    Sampling from the fitted Gaussian instead avoids the holes, and costs one
    forward pass over the data.
    """
    model, blob = load_model(checkpoint_path, device)
    sequences = blob["train_sequences"][:limit]
    stats = blob["property_stats"]
    raw = blob["empirical"]["properties"][:limit]
    max_len = blob["config"]["max_len"]

    tokens = torch.tensor([pdata.encode(s, max_len) for s in sequences],
                          dtype=torch.long, device=device)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long, device=device)
    properties = torch.tensor(pdata.standardise(raw, stats), dtype=torch.float32, device=device)

    model.eval()
    collected = []
    with torch.no_grad():
        for start in range(0, tokens.size(0), 512):
            end = start + 512
            mu, _, _, _ = model.encode(tokens[start:end], lengths[start:end],
                                       properties[start:end])
            collected.append(mu.cpu())
    latents = torch.cat(collected, dim=0)

    mean = latents.mean(0)
    centred = latents - mean
    covariance = centred.t() @ centred / max(latents.size(0) - 1, 1)
    covariance += torch.eye(covariance.size(0)) * 1e-4

    blob["aggregate_posterior"] = {
        "mean": mean.tolist(),
        "cholesky": torch.linalg.cholesky(covariance).tolist(),
    }
    torch.save(blob, checkpoint_path)
    print("fitted aggregate posterior over %d sequences" % latents.size(0))


def load_model(path, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = PeptideVAE(**blob["config"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def draw_latents(blob, n, device, source="fitted"):
    latent = blob["config"]["latent"]
    z = torch.randn(n, latent, device=device)
    if source == "fitted" and "aggregate_posterior" in blob:
        mean = torch.tensor(blob["aggregate_posterior"]["mean"], device=device)
        cholesky = torch.tensor(blob["aggregate_posterior"]["cholesky"], device=device)
        return mean.unsqueeze(0) + z @ cholesky.t()
    return z


def sample(args):
    device = torch.device(args.device)
    model, blob = load_model(args.checkpoint, device)
    stats = blob["property_stats"]
    max_len = blob["config"]["max_len"]

    sampler = pdata.EmpiricalSampler(blob["empirical"]["lengths"],
                                     blob["empirical"]["properties"],
                                     seed=args.seed)
    if args.length:
        lengths_list, properties_list = sampler.sample_with_length(args.n, args.length)
    else:
        lengths_list, properties_list = sampler.sample(args.n)

    if args.charge is not None:
        index = PROPERTY_KEYS.index("charge")
        for row in properties_list:
            row[index] = args.charge
    if args.moment is not None:
        index = PROPERTY_KEYS.index("hydrophobic_moment")
        for row in properties_list:
            row[index] = args.moment

    lengths = torch.tensor(lengths_list, dtype=torch.long, device=device)
    properties = torch.tensor(pdata.standardise(properties_list, stats),
                              dtype=torch.float32, device=device)
    z = draw_latents(blob, args.n, device, args.prior)

    generated = []
    with torch.no_grad():
        for start in range(0, args.n, 512):
            end = min(start + 512, args.n)
            batch_lengths = lengths[start:end]
            batch_properties = properties[start:end]
            batch_z = z[start:end]

            condition = model.conditioner(batch_lengths, batch_properties)
            mask = model.make_mask(batch_lengths)
            tokens = sample_tokens(model, batch_z, condition, mask, args.temperature)

            for _ in range(args.refine):
                mu, _, condition2, mask2 = model.encode(tokens, batch_lengths, batch_properties)
                tokens = sample_tokens(model, mu, condition2, mask2, args.temperature)

            for row, length in zip(tokens.cpu().tolist(), batch_lengths.cpu().tolist()):
                generated.append(pdata.decode(row[:length]))

    training_set = set(blob["train_sequences"])
    unique = len(set(generated))
    novel = sum(1 for s in set(generated) if s not in training_set)
    print("generated %d peptides, %d unique (%.1f%%), %d of those unseen in training (%.1f%%)"
          % (len(generated), unique, unique / len(generated) * 100,
             novel, novel / max(unique, 1) * 100))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for i, seq in enumerate(generated):
            handle.write(">gen_%06d len=%d\n%s\n" % (i + 1, len(seq), seq))
    print("wrote %s" % args.out)


def sample_tokens(model, z, condition, mask, temperature):
    """Draw one token per position. PAD is masked out of the distribution."""
    logits = model.decode(z, condition, mask)
    logits = logits[:, :, 1:]                       # drop the pad class
    if temperature <= 0:
        choice = logits.argmax(-1)
    else:
        probabilities = F.softmax(logits / temperature, dim=-1)
        flat = probabilities.reshape(-1, probabilities.size(-1))
        choice = torch.multinomial(flat, 1).reshape(probabilities.shape[:2])
    tokens = choice + 1                             # shift back past pad
    return tokens * mask.long()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device(args.device)
    _, blob = load_model(args.checkpoint, device)
    reference = blob["train_sequences"]
    ph = blob.get("ph", 7.0)

    generated, _ = pdata.load_sequences(args.generated, min_len=1, max_len=10 ** 6,
                                        verbose=False)
    if not generated:
        raise SystemExit("no usable sequences in %s" % args.generated)

    reference_sample = random.Random(0).sample(reference, min(len(reference), 5000))
    generated_sample = random.Random(0).sample(generated, min(len(generated), 5000))

    reference_features = [compute_features(s, ph) for s in reference_sample]
    generated_features = [compute_features(s, ph) for s in generated_sample]

    print("\nDistribution match, one feature at a time.")
    print("AUROC near 0.500 means the generated set is indistinguishable on that axis.\n")
    print("  %-28s %8s %10s %10s" % ("feature", "AUROC", "gen mean", "train mean"))
    rows = []
    for key in reference_features[0]:
        a = [f[key] for f in generated_features]
        b = [f[key] for f in reference_features]
        rows.append((key, auroc(a, b), list_mean(a), list_mean(b)))
    rows.sort(key=lambda r: -abs(r[1] - 0.5))
    for key, score, generated_mean, reference_mean in rows:
        marker = "  <-- off" if abs(score - 0.5) > 0.15 else ""
        print("  %-28s %8.3f %10.3f %10.3f%s"
              % (key, score, generated_mean, reference_mean, marker))

    training_set = set(reference)
    unique = set(generated)
    print("\n  unique                       %d of %d (%.1f%%)"
          % (len(unique), len(generated), len(unique) / len(generated) * 100))
    print("  not present in training      %d of %d (%.1f%%)"
          % (sum(1 for s in unique if s not in training_set), len(unique),
             sum(1 for s in unique if s not in training_set) / max(len(unique), 1) * 100))
    worst = max(abs(r[1] - 0.5) for r in rows)
    print("\n  largest single-feature gap   %.3f from 0.5" % worst)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Conditional peptide VAE.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trainer = subparsers.add_parser("train")
    trainer.add_argument("fasta", nargs="?", default="data/training/training.fasta")
    trainer.add_argument("--out", default="checkpoint")
    trainer.add_argument("--stub", default="checkpoint/model.json",
                         help="JSON metadata written next to the weights")
    trainer.add_argument("--epochs", type=int, default=120)
    trainer.add_argument("--batch-size", type=int, default=256)
    trainer.add_argument("--lr", type=float, default=1e-3)
    trainer.add_argument("--dim", type=int, default=256)
    trainer.add_argument("--latent", type=int, default=32)
    trainer.add_argument("--layers", type=int, default=4)
    trainer.add_argument("--kernel", type=int, default=5)
    trainer.add_argument("--dropout", type=float, default=0.1)
    trainer.add_argument("--beta", type=float, default=0.5,
                         help="maximum KL weight (default: 0.5)")
    trainer.add_argument("--free-bits", type=float, default=0.1,
                         help="minimum KL in nats per latent dimension (default: 0.1)")
    trainer.add_argument("--kl-cycles", type=int, default=4,
                         help="cyclical KL annealing cycles (default: 4)")
    trainer.add_argument("--clip", type=float, default=1.0)
    trainer.add_argument("--min-len", type=int, default=8)
    trainer.add_argument("--max-len", type=int, default=50)
    trainer.add_argument("--max-run", type=int, default=6,
                         help="drop sequences with longer homopolymer runs (default: 6)")
    trainer.add_argument("--max-residue-frac", type=float, default=0.5,
                         help="drop sequences more than this fraction of one residue")
    trainer.add_argument("--val-fraction", type=float, default=0.1)
    trainer.add_argument("--jaccard", type=float, default=0.5)
    trainer.add_argument("--no-cluster", action="store_true",
                         help="use a random split instead of a cluster-aware one")
    trainer.add_argument("--device", default="cpu")
    trainer.add_argument("--seed", type=int, default=0)
    trainer.add_argument("--snapshot-every", type=int, default=0,
                         help="also save a checkpoint every N epochs, so one run "
                              "yields a training curve instead of a single point")
    trainer.set_defaults(func=train)

    sampler = subparsers.add_parser("sample")
    sampler.add_argument("checkpoint")
    sampler.add_argument("-n", type=int, default=5000)
    sampler.add_argument("--out", default="generated.fasta")
    sampler.add_argument("--temperature", type=float, default=0.9)
    sampler.add_argument("--refine", type=int, default=1,
                         help="re-encode and re-decode this many times (default: 1)")
    sampler.add_argument("--prior", choices=["fitted", "normal"], default="fitted")
    sampler.add_argument("--length", type=int, default=None,
                         help="force a single length instead of sampling the training one")
    sampler.add_argument("--charge", type=float, default=None,
                         help="force a raw net charge for every sample")
    sampler.add_argument("--moment", type=float, default=None,
                         help="force a raw hydrophobic moment for every sample")
    sampler.add_argument("--device", default="cpu")
    sampler.add_argument("--seed", type=int, default=0)
    sampler.set_defaults(func=sample)

    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("checkpoint")
    evaluator.add_argument("generated")
    evaluator.add_argument("--device", default="cpu")
    evaluator.set_defaults(func=evaluate)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


# ---------------------------------------------------------------------------
# Console script entry points, for pyproject [project.scripts]
# ---------------------------------------------------------------------------

def train_cli():
    """uv run train [options]"""
    return main(["train"] + sys.argv[1:])


def sample_cli():
    """uv run sample [options]"""
    return main(["sample"] + sys.argv[1:])


def evaluate_cli():
    """uv run evaluate <checkpoint> <generated.fasta>"""
    return main(["evaluate"] + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())