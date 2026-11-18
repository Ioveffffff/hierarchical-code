"""Parse catalog choices and retrieve lexical alternatives without using labels."""

import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_json_object(raw):
    """Accept a JSON object in plain text or a fenced response, including corrections."""
    if not raw:
        return {}
    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r"\{", str(raw)):
        try:
            value, _ = decoder.raw_decode(str(raw)[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects[-1] if objects else {}


def catalog_candidates(terms, representatives, catalog, limit=15):
    """Rank every catalog name by character n-gram overlap with cluster terms."""
    if not terms or not catalog:
        raise ValueError("类别检索需要术语和类别清单")
    labels = sorted(catalog)
    documents = [catalog[label] for label in labels]
    queries = [" ".join(dict.fromkeys(terms)),
               " ".join(dict.fromkeys(representatives or terms))]
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                             sublinear_tf=True).fit_transform(documents + queries)
    score = (matrix[:len(labels)] @ (
        0.8 * matrix[len(labels)] + 0.2 * matrix[len(labels) + 1]).T).toarray().ravel()
    order = sorted(range(len(labels)), key=lambda i: (-score[i], labels[i]))
    return [{"label": labels[i], "name": catalog[labels[i]],
             "lexical_score": round(float(score[i]), 4)}
            for i in order[:limit]]
