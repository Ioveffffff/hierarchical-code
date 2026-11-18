"""Unsupervised BSE instance contrast using independent dropout views."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def instance_nt_xent(features, texts, temperature=0.1):
    """Each text's two views are positives; other identical texts are masked.

    ``features`` is ordered [view A for the batch, view B for the batch].
    No category labels or pseudo labels enter this objective.
    """
    batch_size = len(texts)
    if batch_size < 2 or features.shape[0] != 2 * batch_size:
        raise ValueError('Need at least two texts and exactly two views per text')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    positions = torch.arange(2 * batch_size, device=features.device)
    positives = (positions + batch_size) % (2 * batch_size)
    # Equal text at another row is not a valid negative, even if its row ID differs.
    duplicate = torch.as_tensor(
        [[a == b for b in texts * 2] for a in texts * 2],
        dtype=torch.bool, device=features.device)
    duplicate[positions, positives] = False
    duplicate[positions, positions] = True
    logits = logits.masked_fill(duplicate, torch.finfo(logits.dtype).min)
    return F.cross_entropy(logits, positives)


def pretrain_bse_instance(
        model, entities, epochs=100, batch_size=48,
        lr_backbone=1e-6, lr_other=5e-5, device='cuda', seed=42,
        temperature=0.1):
    """Train the BSE representation with two stochastic views of each text."""
    if len(entities) < 2 or batch_size < 2 or epochs < 1:
        raise ValueError('Need >=2 entities, batch_size>=2, and epochs>=1')
    if model.cluster_projector is not None:
        raise ValueError('Unsupervised BSE must not have a classifier head')
    groups = [
        dict(params=[p for n, p in model.named_parameters()
                     if p.requires_grad and n.startswith('backbone.')], lr=lr_backbone),
        dict(params=[p for n, p in model.named_parameters()
                     if p.requires_grad and not n.startswith('backbone.')], lr=lr_other),
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(len(entities))
        batches = [permutation[i:i + batch_size]
                   for i in range(0, len(permutation), batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2] = np.concatenate([batches[-2], batches[-1]])
            batches.pop()
        weighted_loss = 0.0
        total_count = 0
        for indices in batches:
            texts = [entities[int(i)] for i in indices]
            # The backbone, character CNN and projections draw independent
            # dropout masks for the two copies in one training-mode forward.
            features = model(texts + texts, return_all=True)['z_normalized']
            loss = instance_nt_xent(features, texts, temperature)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite BSE loss at epoch {epoch + 1}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            weighted_loss += float(loss.detach()) * len(indices)
            total_count += len(indices)
        record = dict(epoch=epoch + 1, loss=weighted_loss / total_count,
                      n_entities=total_count, objective='instance_nt_xent')
        history.append(record)
        print('Unsupervised BSE: ' + json.dumps(record), flush=True)
    return history


def save_bse_history(output_dir, history):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / 'bse_instance_history.json'
    target.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(target)
