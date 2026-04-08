# CaSaMamba

Current status: this repository is initialized with **B0 only** (a migrated baseline of Mamba6mA).

## Scope

- Implemented now: **B0** (structure migration, baseline behavior preserved)
- Not implemented yet: **B1** (Mamba-2 backbone), **B2** (center-aware pooling), **B3** (adaptive-scale fusion)
- B0 target in this stage: align behavior with original `Mamba6mA` as closely as possible.

## Data

This project now reads data from its own directory:

- `./data/`

For B0, copy these dataset folders into `CaSaMamba/data/`:

- `6mA_A.thaliana`
- `6mA_C.elegans`
- `6mA_H.sapiens`
- `6mA_R.chinensis`

Each folder should contain:

- `train_neg.txt`
- `train_pos.txt`
- `test_neg.txt`
- `test_pos.txt`

## Run B0

Install dependencies (PyTorch is expected to be preinstalled):

```bash
pip install -r requirements.txt
```

Train baseline (same default as original `Mamba6mA/train.py`, dataset `6mA_C.elegans`):

```bash
python train.py
```

Predict logits with original-style default entry (`predict.py` keeps original defaults):

```bash
python predict.py
```

For a local train->predict closed loop on one dataset (`6mA_C.elegans`):

```bash
bash scripts/train_b0.sh
bash scripts/predict_b0.sh
```

Default prediction output:

- `result/output_logits.xlsx`

## Planned stages

- **B1**: replace baseline Mamba block with Mamba-2 style backbone
- **B2**: add center-aware pooling on top of B1
- **B3**: add adaptive-scale fusion on top of B2
