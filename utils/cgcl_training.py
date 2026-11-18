"""Unsupervised three-stage CGCL; ground-truth labels are not an input."""
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import silhouette_score

from clustering_utils import pseudo_label_change
from utils.cgcl_pseudo import (choose_partition, pseudo_partition,
                               effective_training_labels, smooth_teacher,
                               inject_pseudo_noise)
from utils.loss import ImprovedSupConLoss, WhiteningLoss, HierarchicalContrastiveLoss
from utils.improved_hierarchy import augment_entity


@torch.no_grad()
def extract_cgcl_embeddings(model, entities, batch_size=64):
    was_training = model.training
    model.eval()
    try:
        return torch.cat([model(entities[i:i+batch_size]).detach().cpu()
                          for i in range(0, len(entities), batch_size)]).numpy()
    finally:
        model.train(was_training)


def make_batches(n, batch_size, rng, fine=None, coarse=None):
    """Cover every entity, mixing fine pairs across the entire dataset.

    Global shuffling avoids batches containing almost exclusively one coarse
    cluster. Two views provide positives even for untrusted/singleton entities.
    """
    if fine is None:
        order = rng.permutation(n).tolist()
    else:
        pairs = []
        for label in np.unique(fine):
            indices = rng.permutation(np.flatnonzero(fine == label)).tolist()
            pairs.extend(indices[i:i+2] for i in range(0, len(indices), 2))
        rng.shuffle(pairs)
        order = [i for pair in pairs for i in pair]
    batches = [order[i:i+batch_size] for i in range(0, n, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    rng.shuffle(batches)
    return batches


def fixed_probe(entities, batch_size, seed, limit=256):
    """Fixed label-free monitoring examples, not a held-out validation set."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(entities)), min(limit, len(entities)))
    def view(text):
        text = str(text)
        if len(text) > 3:
            i = rng.randrange(len(text))
            return text[:i] + text[i+1:]
        return text
    texts = [entities[i] for i in indices]
    views1, views2 = [view(t) for t in texts], [view(t) for t in texts]
    return [(views1[i:i+batch_size], views2[i:i+batch_size])
            for i in range(0, len(texts), batch_size)]


@torch.no_grad()
def probe_score(model, probe, instance_loss, device):
    was_training = model.training
    model.eval()
    weighted, count = 0.0, 0
    try:
        for a, b in probe:
            if len(a) < 2:
                continue
            out = model(a + b, return_all=True)
            labels = torch.arange(len(a), device=device).repeat(2)
            weighted += float(instance_loss(out['z_normalized'], labels)) * len(a)
            count += len(a)
        if not count:
            raise ValueError('No valid monitoring batch')
        return weighted / count
    finally:
        model.train(was_training)


def train_three_stage_improved(
        model, entities, epochs_stage1=20, epochs_stage2=100, epochs_stage3=90,
        batch_size=48, lr_backbone=1e-6, lr_other=5e-5, temperature=0.07,
        alpha=0.1, whitening_weight=0.1, device='cuda',
        pseudo_update_interval=5, cluster_k_min=2, cluster_k_max=256,
        cluster_sample_size=1024, target_coarse_fine_ratio=0.3,
        confidence_margin=0.02, history_weight=0.5, anchor_weight=0.2,
        std_target=0.1, hard_margin=0.2, hard_weight=0.2, warmup_epochs=20,
        anchor_distill_weight=0.5, monitor_interval=5, output_root='./outputs', seed=42,
        output_dir=None, timestamp=None, fixed_k=None,
        final_embedding_source='cgcl', use_coarse_contrast=True,
        use_fine_contrast=True, dynamic_pseudo_labels=True,
        stage2_fine_weight=0.0, pseudo_noise_rate=0.0,
        pseudo_noise_scope='fine'):
    if len(entities) < 4 or batch_size < 2:
        raise ValueError('Need >=4 entities and batch_size>=2')
    epochs = [epochs_stage1, epochs_stage2, epochs_stage3]
    if epochs_stage1 < 0 or epochs_stage2 < 1 or epochs_stage3 < 1:
        raise ValueError('Stage1 may be skipped; stages 2 and 3 need >=1 epoch')
    if pseudo_update_interval < 1 or monitor_interval < 1:
        raise ValueError('Update/monitor intervals must be positive')
    if not 0 < target_coarse_fine_ratio < 1:
        raise ValueError('Invalid coarse/fine target ratio')
    if not 0 <= history_weight < 1 or not 0 <= anchor_weight < 1:
        raise ValueError('Smoothing weights must be in [0, 1)')
    if cluster_k_min < 2 or cluster_k_max < cluster_k_min or cluster_sample_size < 4:
        raise ValueError('Invalid cluster search configuration')
    if fixed_k is not None and not 2 <= int(fixed_k) <= len(entities) // 2:
        raise ValueError('fixed_k must be between 2 and half the sample count')
    if final_embedding_source not in ('bse_anchor', 'cgcl'):
        raise ValueError('final_embedding_source must be bse_anchor or cgcl')
    if not 0 <= alpha <= 1 or confidence_margin < 0 or whitening_weight < 0 or anchor_distill_weight < 0:
        raise ValueError('Invalid loss/confidence parameters')
    if not (use_coarse_contrast or use_fine_contrast):
        raise ValueError('At least one contrastive granularity must be enabled')
    if not 0 <= stage2_fine_weight <= 1 or (stage2_fine_weight and
                                              not (use_coarse_contrast and use_fine_contrast)):
        raise ValueError('Stage2 fine weight requires both granularities and must be in [0, 1]')
    if not 0 <= pseudo_noise_rate <= 1 or pseudo_noise_scope not in ('fine', 'coarse'):
        raise ValueError('Invalid pseudo-label noise rate/scope')
    config = {k: v for k, v in locals().items() if k not in ('model', 'entities')}
    config['method'] = 'unsupervised_instance_bse_then_pseudo_label_cgcl'
    config['n_entities'] = len(entities)
    config['fixed_k_prespecified'] = fixed_k
    config['final_embedding_source'] = final_embedding_source
    config['ground_truth_labels_used_for_training'] = False
    config['stage_objectives'] = {
        'stage1': ('skipped after unsupervised BSE instance training'
                   if epochs_stage1 == 0 else 'label-free instance contrast'),
        'stage2': ('coarse/fine pseudo-label contrast' if stage2_fine_weight else
                   'coarse pseudo-label contrast' if use_coarse_contrast else
                   'instance contrast without coarse pseudo labels'),
        'stage3': ('coarse and fine pseudo-label contrast' if use_coarse_contrast and use_fine_contrast
                   else 'coarse-only contrast' if use_coarse_contrast else 'fine-only contrast'),
    }
    config['selection'] = (
        'S1 skipped after unsupervised BSE training; S2 coarse/S3 fine cosine silhouette; maximize'
        if epochs_stage1 == 0 else
        'S1 negative instance loss; S2 coarse/S3 fine cosine silhouette; maximize')
    timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = Path(output_dir) if output_dir is not None else Path(output_root) / timestamp
    output.mkdir(parents=True, exist_ok=output_dir is not None)
    (output / 'global_config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'entities.json').write_text(
        json.dumps(entities, ensure_ascii=False, indent=2), encoding='utf-8')
    groups = [dict(params=[p for n, p in model.named_parameters()
                           if p.requires_grad and n.startswith('backbone.')], lr=lr_backbone),
              dict(params=[p for n, p in model.named_parameters()
                           if p.requires_grad and not n.startswith('backbone.')], lr=lr_other)]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    instance_loss = ImprovedSupConLoss(temperature=temperature)
    regularizer = WhiteningLoss(std_target=std_target).to(device)
    hierarchy = HierarchicalContrastiveLoss(
        temperature=temperature,
        coarse_weight=(alpha if use_fine_contrast else 1.0) if use_coarse_contrast else 0.0,
        fine_weight=(1-alpha if use_coarse_contrast else 1.0) if use_fine_contrast else 0.0,
        hard_margin=hard_margin,
        hard_weight=hard_weight if use_coarse_contrast and use_fine_contrast else 0.0)
    probe = fixed_probe(entities, batch_size, seed)
    # Preserve the unsupervised BSE geometry while pseudo-labels are still noisy.
    # This uses frozen features only; true labels never enter CGCL.
    initial_embeddings = extract_cgcl_embeddings(model, entities, batch_size)
    np.save(output / 'bse_anchor_embeddings.npy', initial_embeddings)
    initial_state = {name: value.detach().cpu().clone()
                     for name, value in model.state_dict().items()}
    torch.save(initial_state, output / 'bse_anchor_model.pt')
    search = dict(k_min=cluster_k_min, k_max=cluster_k_max,
                  sample_size=cluster_sample_size, seed=seed)
    partition_options = dict(**search, fixed_k=fixed_k)
    anchor_coarse, anchor_fine, anchor_reliable, _ = pseudo_partition(
        initial_embeddings, coarse_ratio=target_coarse_fine_ratio,
        confidence_margin=confidence_margin, **partition_options)
    np.savez(output / 'bse_anchor_unsupervised.npz', embeddings=initial_embeddings,
             fine=anchor_fine, coarse=anchor_coarse, reliable=anchor_reliable)
    coarse = fine = reliable = teacher = anchor = frozen_labels = None
    update_count = global_epoch = 0

    def refresh(stage, epoch):
        nonlocal coarse, fine, reliable, teacher, anchor, frozen_labels, update_count
        embeddings = extract_cgcl_embeddings(model, entities, batch_size)
        if anchor is None:
            anchor = embeddings.copy()
        teacher = smooth_teacher(embeddings, teacher, anchor, history_weight, anchor_weight)
        clean_coarse, clean_fine, reliable, stats = pseudo_partition(
            teacher, coarse_ratio=target_coarse_fine_ratio,
            confidence_margin=confidence_margin, **partition_options)
        new_coarse, new_fine, noise_mask, noise_info = inject_pseudo_noise(
            clean_coarse, clean_fine, reliable, rate=pseudo_noise_rate,
            scope=pseudo_noise_scope, seed=seed + 1009*update_count)
        fine_sizes = np.unique(new_fine[reliable], return_counts=True)[1]
        coarse_sizes = np.unique(new_coarse[reliable], return_counts=True)[1]
        pairs = lambda sizes: int(np.sum(sizes*(sizes-1)//2))
        nested = sum(len(np.unique(new_coarse[new_fine == label])) == 1
                     for label in np.unique(new_fine))
        stats.update(noise_info)
        stats.update(stage=stage, epoch=epoch, global_epoch=global_epoch,
                     fine_change=pseudo_label_change(fine, new_fine),
                     coarse_change=pseudo_label_change(coarse, new_coarse),
                     fine_coarse_nested_fraction=nested/len(np.unique(new_fine)),
                     reliable_fine_positive_pairs=pairs(fine_sizes),
                     reliable_same_coarse_different_fine_pairs=(pairs(coarse_sizes)-pairs(fine_sizes)))
        coarse, fine = new_coarse, new_fine
        if update_count == 0 and not dynamic_pseudo_labels:
            frozen_labels = (coarse.copy(), fine.copy(), reliable.copy())
        update_count += 1
        np.savez(output / f'pseudo_{update_count:03d}.npz', fine=fine, coarse=coarse,
                 clean_fine=clean_fine, clean_coarse=clean_coarse, noise_mask=noise_mask,
                 reliable=reliable, embeddings=embeddings, teacher_embeddings=teacher)
        with (output / 'pseudo_updates.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps(stats, ensure_ascii=False) + '\n')
        print(f'Pseudo update {update_count}: ' + json.dumps(
            {k: v for k, v in stats.items() if k != 'candidates'}), flush=True)

    def monitor(stage):
        if stage == 1:
            score = -probe_score(model, probe, instance_loss, device)
            return dict(selection_score=score, selection_metric='negative_instance_loss')
        if stage == 2 and not use_coarse_contrast:
            score = -probe_score(model, probe, instance_loss, device)
            return dict(selection_score=score, selection_metric='negative_instance_loss')
        embeddings = extract_cgcl_embeddings(model, entities, batch_size)
        if stage == 2 or not use_fine_contrast:
            monitor_coarse, _, _, info = pseudo_partition(
                embeddings, coarse_ratio=target_coarse_fine_ratio,
                confidence_margin=confidence_margin, **partition_options)
            indices = np.sort(np.random.default_rng(seed).choice(
                len(embeddings), min(len(embeddings), cluster_sample_size), replace=False))
            sampled_labels = monitor_coarse[indices]
            n_sampled_clusters = len(np.unique(sampled_labels))
            if 2 <= n_sampled_clusters < len(indices):
                score = float(silhouette_score(
                    embeddings[indices], sampled_labels, metric='cosine'))
                metric = 'coarse_cosine_silhouette'
                selected_k = int(len(np.unique(monitor_coarse)))
            else:
                score = float(info['silhouette'])
                metric = 'fine_cosine_silhouette_fallback'
                selected_k = int(info['selected_k'])
            # Keep checkpoint selection independent of the Stage2 loss weight.
            # Both baseline and fine-weight ablations use the same coarse score.
            return dict(selection_score=score, selection_metric=metric,
                        selected_k=selected_k, selected_fine_k=info['selected_k'],
                        partition=info)
        if fixed_k is None:
            _, info, _ = choose_partition(embeddings, **search)
        else:
            _, _, _, info = pseudo_partition(
                embeddings, coarse_ratio=target_coarse_fine_ratio,
                confidence_margin=confidence_margin, **partition_options)
        return dict(selection_score=info['silhouette'], selection_metric='fine_cosine_silhouette',
                    selected_k=info['selected_k'], partition=info)

    def save_checkpoint(path, stage, epoch, selection):
        torch.save(dict(model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), epoch=epoch, stage=stage,
                        config=config, coarse_labels=coarse, fine_labels=fine,
                        reliable=reliable, teacher_embeddings=teacher, **selection), path)

    for stage, n_epochs in enumerate(epochs, start=1):
        stage_dir = output / f'stage{stage}'
        stage_dir.mkdir()
        optimizer.state.clear()
        best_score = -float('inf')
        history = []
        # Retain the incoming model if this stage harms its clustering geometry.
        if stage > 1:
            selection = monitor(stage)
            best_score = selection['selection_score']
            save_checkpoint(stage_dir / 'best.pt', stage, 0, selection)
            (stage_dir / 'selection.jsonl').write_text(
                json.dumps(dict(epoch=0, global_epoch=global_epoch, **selection)) + '\n',
                encoding='utf-8')
        elif n_epochs == 0:
            selection = monitor(stage)
            best_score = selection['selection_score']
            save_checkpoint(stage_dir / 'best.pt', stage, 0, selection)
            (stage_dir / 'selection.jsonl').write_text(
                json.dumps(dict(epoch=0, global_epoch=global_epoch, **selection)) + '\n',
                encoding='utf-8')
        for epoch in range(n_epochs):
            if stage > 1 and ((dynamic_pseudo_labels and epoch % pseudo_update_interval == 0)
                              or (not dynamic_pseudo_labels and update_count == 0)):
                refresh(stage, epoch)
            model.train()
            rng = np.random.default_rng(seed + global_epoch)
            train_coarse, train_fine = (effective_training_labels(coarse, fine, reliable)
                                        if stage > 1 else (None, None))
            pair_labels = (train_coarse if use_coarse_contrast else None) if stage == 2 else train_fine
            batches = make_batches(len(entities), batch_size, rng, pair_labels, train_coarse)
            sums = np.zeros(10)
            count = 0
            for indices in tqdm(batches, desc=f'CGCL S{stage} {epoch+1}/{n_epochs}', ncols=100):
                texts = [entities[i] for i in indices]
                probability = 0.15 if stage == 1 else 0.10
                views = [augment_entity(t, probability) for t in texts]
                views += [augment_entity(t, probability) for t in texts]
                out = model(views, return_all=True)
                features, raw = out['z_normalized'], out['z']
                if stage == 1:
                    labels = torch.arange(len(indices), device=device).repeat(2)
                    main = instance_loss(features, labels)
                    coarse_loss = fine_loss = hard_loss = main.detach() * 0
                else:
                    batch_fine = torch.as_tensor(train_fine[indices], device=device).repeat(2)
                    if stage == 2:
                        if use_coarse_contrast:
                            batch_coarse = torch.as_tensor(train_coarse[indices], device=device).repeat(2)
                            coarse_loss = instance_loss(features, batch_coarse)
                            fine_loss = (instance_loss(features, batch_fine)
                                         if stage2_fine_weight else coarse_loss.detach() * 0)
                            main = ((1-stage2_fine_weight)*coarse_loss
                                    + stage2_fine_weight*fine_loss)
                        else:
                            instance_labels = torch.arange(len(indices), device=device).repeat(2)
                            main = instance_loss(features, instance_labels)
                            coarse_loss = fine_loss = main.detach() * 0
                        hard_loss = main.detach() * 0
                    else:
                        batch_coarse = torch.as_tensor(train_coarse[indices], device=device).repeat(2)
                        hierarchical, coarse_loss, fine_loss = hierarchy(features, batch_coarse, batch_fine)
                        hard_loss = (hierarchy.hard_negative_loss(features, batch_coarse, batch_fine)
                                     if hierarchy.hard_weight else hierarchical.detach() * 0)
                        if use_fine_contrast:
                            weight = min(1.0, (epoch+1) / max(1, warmup_epochs))
                            main = (1-weight)*fine_loss + weight*hierarchical
                        else:
                            main = hierarchical
                    anchor_target = torch.as_tensor(
                        initial_embeddings[indices], dtype=features.dtype, device=device).repeat(2, 1)
                    anchor_loss = (1.0 - F.cosine_similarity(features, anchor_target, dim=1)).mean()
                variance, covariance, raw_std, std_fraction = regularizer.components(raw)
                white = (variance + regularizer.covariance_weight * covariance
                         if stage > 1 else raw.sum()*0.0)
                if stage == 1:
                    anchor_loss = main.detach() * 0
                loss = main + whitening_weight * white + anchor_distill_weight * anchor_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss in stage {stage}, epoch {epoch}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                sums += np.array([float(v.detach()) for v in
                    (loss, white, coarse_loss, fine_loss, hard_loss, anchor_loss,
                     variance, covariance, raw_std, std_fraction)]) * len(indices)
                count += len(indices)
            global_epoch += 1
            record = dict(epoch=epoch+1, global_epoch=global_epoch,
                          **dict(zip(('loss', 'regularizer', 'coarse_loss', 'fine_loss', 'hard_loss', 'anchor_loss',
                                      'variance_penalty', 'covariance_penalty', 'raw_std',
                                      'std_target_fraction'), (sums/count).tolist())))
            if (epoch+1) % monitor_interval == 0 or epoch == 0 or epoch+1 == n_epochs:
                selection = monitor(stage)
                score = selection['selection_score']
                if not np.isfinite(score):
                    raise RuntimeError('Nonfinite monitoring score')
                record.update({k: v for k, v in selection.items() if k != 'partition'})
                with (stage_dir / 'selection.jsonl').open('a', encoding='utf-8') as file:
                    file.write(json.dumps(dict(epoch=epoch+1, global_epoch=global_epoch, **selection)) + '\n')
                if score > best_score:
                    best_score = score
                    save_checkpoint(stage_dir / 'best.pt', stage, epoch+1, selection)
            history.append(record)
            print(json.dumps(record), flush=True)
            (stage_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        save_checkpoint(stage_dir / 'final.pt', stage, n_epochs, {})
        # Keep an explicit empty history for a skipped Stage1.
        (stage_dir / 'history.json').write_text(
            json.dumps(history, indent=2), encoding='utf-8')
        best = torch.load(stage_dir / 'best.pt', map_location=device, weights_only=False)
        model.load_state_dict(best['model_state_dict'])
        # Restore the pseudo-label state paired with the selected model checkpoint.
        # Resetting the teacher here can change the cut even when embeddings are identical.
        coarse, fine, reliable = best['coarse_labels'], best['fine_labels'], best['reliable']
        if not dynamic_pseudo_labels and frozen_labels is not None:
            coarse, fine, reliable = frozen_labels
        teacher = best.get('teacher_embeddings')
        print(f'Stage {stage}: loaded best epoch {best["epoch"]}, score={best["selection_score"]:.6f} (maximize), teacher_restored={teacher is not None}')
    cgcl_embeddings = extract_cgcl_embeddings(model, entities, batch_size)
    cgcl_coarse, cgcl_fine, cgcl_reliable, cgcl_info = pseudo_partition(
        cgcl_embeddings, coarse_ratio=target_coarse_fine_ratio,
        confidence_margin=confidence_margin, **partition_options)
    np.savez(output / 'cgcl_final_unsupervised.npz', embeddings=cgcl_embeddings,
             fine=cgcl_fine, coarse=cgcl_coarse, reliable=cgcl_reliable)
    if final_embedding_source == 'bse_anchor':
        embeddings = initial_embeddings
        model.load_state_dict(initial_state)
    else:
        embeddings = cgcl_embeddings
    coarse, fine, reliable, info = pseudo_partition(
        embeddings, coarse_ratio=target_coarse_fine_ratio,
        confidence_margin=confidence_margin, **partition_options)
    info['embedding_source'] = final_embedding_source
    np.savez(output / 'final_unsupervised.npz', embeddings=embeddings,
             fine=fine, coarse=coarse, reliable=reliable)
    torch.save(model.state_dict(), output / 'final_model.pt')
    (output / 'final_cluster_selection.json').write_text(
        json.dumps(info, indent=2), encoding='utf-8')
    return str(output), timestamp
