"""Encode concept names for the second Leiden similarity view."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests


class ConceptNameEncoder:
    def __init__(self, chat_url, api_key, cache_dir, model="text-embedding-v4"):
        if not chat_url or not api_key:
            raise ValueError("名称嵌入需要已配置的 API")
        self.endpoint = chat_url.rsplit("/chat/completions", 1)[0] + "/embeddings"
        self.api_key = api_key
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory = {}

    def encode(self, names):
        names = [str(name).strip() for name in names]
        if not names or any(not name for name in names):
            raise ValueError("概念名称不能为空")
        missing = list(dict.fromkeys(name for name in names if name not in self.memory))
        for name in missing:
            digest = hashlib.sha256(json.dumps([self.model, name],
                                             ensure_ascii=False).encode()).hexdigest()
            cache = self.cache_dir / f"{digest}.npy"
            if cache.exists():
                self.memory[name] = np.load(cache, allow_pickle=False)
        missing = [name for name in missing if name not in self.memory]
        for offset in range(0, len(missing), 10):
            batch = missing[offset:offset + 10]
            for attempt in range(4):
                try:
                    response = requests.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}",
                                 "Content-Type": "application/json"},
                        json={"model": self.model, "input": batch}, timeout=90)
                    response.raise_for_status()
                    records = sorted(response.json()["data"],
                                     key=lambda row: row["index"])
                    if len(records) != len(batch):
                        raise ValueError("名称嵌入 API 返回数量不完整")
                    for name, record in zip(batch, records):
                        vector = np.asarray(record["embedding"], dtype=np.float64)
                        norm = np.linalg.norm(vector)
                        if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= 1e-12:
                            raise ValueError("名称嵌入无效")
                        vector = vector / norm
                        self.memory[name] = vector
                        digest = hashlib.sha256(json.dumps([self.model, name],
                                                         ensure_ascii=False).encode()).hexdigest()
                        np.save(self.cache_dir / f"{digest}.npy", vector)
                    break
                except (requests.RequestException, KeyError, ValueError) as exc:
                    if attempt == 3:
                        raise RuntimeError(f"名称嵌入失败，起始位置 {offset}") from exc
                    time.sleep(2 ** attempt)
        matrix = np.vstack([self.memory[name] for name in names])
        if len({len(vector) for vector in self.memory.values()}) != 1:
            raise ValueError("名称嵌入维度不一致")
        return matrix
