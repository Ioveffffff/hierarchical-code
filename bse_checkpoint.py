"""Auditable shared unsupervised-BSE starting point for paired CGCL runs."""
import hashlib
import json
from pathlib import Path

import torch

from load_dataset import load_texts_from_csv

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ('main_ACRL_可视化.py', 'unsupervised_bse.py', 'dual_channel_model.py',
                'load_dataset.py', 'bse_checkpoint.py')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bse_cache_key(csv, backbone, seed, epochs_bse, batch_size):
    """Cache identity depends on texts, settings and code, never category labels."""
    texts = load_texts_from_csv(csv)
    text_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode('utf-8')).hexdigest()
    inputs = dict(training_text_sha256=text_hash, backbone=str(Path(backbone).resolve()),
                  seed=int(seed), epochs_bse=int(epochs_bse),
                  batch_size=int(batch_size),
                  training_mode='unsupervised_instance_nt_xent_v1',
                  source_sha256={name: sha256_file(ROOT / name) for name in SOURCE_FILES})
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode('utf-8')).hexdigest()
    return key, inputs


def expected_metadata(csv, backbone, seed, epochs_bse, batch_size,
                      n_entities, vocab_size):
    key, inputs = bse_cache_key(csv, backbone, seed, epochs_bse, batch_size)
    return dict(cache_key=key, **inputs, n_entities=int(n_entities),
                vocab_size=int(vocab_size))


def save_bse_checkpoint(path, model, metadata, history):
    """Write once; an existing cached starting point must never be replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'BSE checkpoint already exists: {path}')
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise FileExistsError(f'Incomplete BSE checkpoint exists: {temporary}')
    try:
        torch.save(dict(model_state_dict=model.state_dict(), metadata=metadata,
                        instance_history=history), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def load_bse_checkpoint(path, model, expected):
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('metadata') != expected:
        raise ValueError('BSE checkpoint data, seed, architecture, or pretraining settings differ')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return checkpoint['instance_history'], sha256_file(path)
