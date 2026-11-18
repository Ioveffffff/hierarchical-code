"""Durable LLM decisions for one hierarchy layer and one frozen input."""

import hashlib
import json
from pathlib import Path

import numpy as np


class SemanticReviewCache:
    def __init__(self, path, vectors, names, examples, flags, layer, model):
        self.path = Path(path)
        digest = hashlib.sha256(np.asarray(vectors, dtype=np.float64).tobytes())
        digest.update(json.dumps([names, examples, flags, layer, model, "v3"],
                                 ensure_ascii=False).encode())
        self.signature = digest.hexdigest()
        self.groups = {}
        self.edges = {}
        self.collisions = {}
        self.collision_edges = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("signature") != self.signature:
                    raise ValueError(f"语义审核断点与当前层输入不一致：{self.path}")
                if row["kind"] == "group":
                    self.groups[tuple(row["child_ids"])] = row["decision"]
                elif row["kind"] == "edge":
                    self.edges[(row["child_id"], row["parent_name"])] = row["score"]
                elif row["kind"] == "collision":
                    key = (tuple(row["previous_child_ids"]),
                           tuple(row["current_child_ids"]), row["name"],
                           tuple(row["occupied_names"]))
                    self.collisions[key] = row["decision"]
                elif row["kind"] == "collision_edge":
                    self.collision_edges[(row["child_id"], row["parent_name"])] = row["score"]

    def _append(self, row):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"signature": self.signature, **row},
                                    ensure_ascii=False) + "\n")
            stream.flush()

    def record_group(self, ids, decision):
        ids = tuple(ids)
        self.groups[ids] = decision
        self._append({"kind": "group", "child_ids": ids, "decision": decision})

    def record_edge(self, child_id, parent_name, score):
        self.edges[(child_id, parent_name)] = score
        self._append({"kind": "edge", "child_id": child_id,
                      "parent_name": parent_name, "score": score})

    def record_collision(self, previous_ids, current_ids, name, occupied, decision):
        key = (tuple(previous_ids), tuple(current_ids), name,
               tuple(sorted(occupied)))
        self.collisions[key] = decision
        self._append({"kind": "collision", "previous_child_ids": key[0],
                      "current_child_ids": key[1], "name": name,
                      "occupied_names": key[3], "decision": decision})

    def record_collision_edge(self, child_id, parent_name, score):
        self.collision_edges[(child_id, parent_name)] = score
        self._append({"kind": "collision_edge", "child_id": child_id,
                      "parent_name": parent_name, "score": score})
