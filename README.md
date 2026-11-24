# Hierarchical Clustering Research Code

This repository contains the Python implementation of unsupervised BSE instance
contrastive pretraining, CGCL pseudo-label training, and HCRL hierarchy building.

## BSE and CGCL

Install `requirements.txt` and a PyTorch build compatible with the target
machine. `main_ACRL_可视化.py` trains BSE and CGCL from a CSV containing
`label,text` or `label,text0,text1`, and a local Transformer checkpoint:

```bash
python main_ACRL_可视化.py --csv /path/to/dataset.csv --backbone /path/to/checkpoint
```

The training stages read only the text columns. Ground-truth labels are loaded
after training to report evaluation metrics.

## Datasets

The CSV files in `datasets/` use UTF-8 and `label,text` columns:

- `THUCNews.csv`: 10,000 Chinese news records across ten labels.
- `motor_sample_dataset_.csv`: 28 motor terminology records across four labels.
  This is an illustrative subset; the full enterprise dataset is not public.
- `test.csv`: a smaller example of the same CSV format.

The repository does not include the full motor dataset.

## Full pipeline

`run_full_baseline.py` orchestrates BSE, CGCL, and HCRL for the full dataset. It
expects the dataset, the L1 category catalog, a local RoBERTa checkpoint, the
LLM API credential, and supporting experiment scripts to be supplied for a run.

Before running HCRL on a dataset, fill the `PROMPT_*` dataset-specific values in
`utils/hierarchical_knowledge_builder.py` and provide the Leiden settings via
`HCRL_LEIDEN_NEIGHBORS`, `HCRL_LEIDEN_RESOLUTIONS`, and `HCRL_LEIDEN_SEEDS`.
Do not commit API credentials, model checkpoints, generated outputs, or private
full datasets.

No license is included.
