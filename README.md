# Hierarchical Clustering Research Code

This repository contains the Python implementation of unsupervised BSE instance
contrastive pretraining, CGCL pseudo-label training, and HCRL hierarchy building.

## Quick smoke test

The included `datasets/test.csv` is a small execution test set. It contains a
`label,text` header and is used for evaluation only; BSE and CGCL training read
the text column without using labels.

Install the dependencies in `requirements.txt`, install a PyTorch build that
matches the machine's CUDA setup, then run:

```bash
python -B run_smoke_test.py --device cuda:0
```

To use a different local RoBERTa checkpoint:

```bash
python -B run_smoke_test.py --device cuda:0 --backbone /path/to/chinese-roberta-wwm-ext
```

The smoke test runs one BSE epoch and one epoch in each active CGCL stage. It
does not run HCRL and is not intended to produce research metrics.

## Full pipeline

`run_full_baseline.py` orchestrates BSE, CGCL, and HCRL for the full dataset. It
expects the dataset, the L1 category catalog, a local RoBERTa checkpoint, and
the LLM API credential to be supplied outside the repository.

Before running HCRL on a dataset, fill the `PROMPT_*` dataset-specific values in
`utils/hierarchical_knowledge_builder.py` and provide the Leiden settings via
`HCRL_LEIDEN_NEIGHBORS`, `HCRL_LEIDEN_RESOLUTIONS`, and `HCRL_LEIDEN_SEEDS`.
Do not commit API credentials, model checkpoints, generated outputs, or private
full datasets.

No license is included. Add one only after deciding how the code may be reused.
