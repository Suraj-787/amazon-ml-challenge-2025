# Smart Product Pricing — Amazon ML Challenge 2025

Predicting product prices from catalog text and product images, using a fine-tuned DeBERTa-v3 text model and a CLIP + LoRA image model.

**Team Monk Builders:** Suraj Kumar, Himanshu Rai, Pratik Gautam, Impana KS

## Problem

Each product comes with free-form catalog content (item name, bullet points, description, and a value/unit such as `Value: 72.0 Unit: Fl Oz`) and an image link. The task is to predict its price.

| Split | Rows | Columns |
|---|---|---|
| Train | 75,000 | `sample_id`, `catalog_content`, `image_link`, `price` |
| Test | 75,000 | `sample_id`, `catalog_content`, `image_link` |

Submissions are scored with **SMAPE** (Symmetric Mean Absolute Percentage Error) — lower is better.

> The competition dataset is not included in this repository. Place `train.csv` and `test.csv` in `data/` before running anything.

## Approach

The work went through three stages. The process docs for each stage are in [`docs/`](docs/).

### 1. Baseline: DeBERTa-v3-large regressor

- Fine-tuned `microsoft/deberta-v3-large` on cleaned catalog text (HTML removal, unicode normalisation, whitespace cleaning), max length 256.
- 3-layer regression head (1024 → 512 → 256 → 1, GELU, dropout 0.1).
- Trained on `log(price + 1)` with MSE loss and differential learning rates (encoder 1e-5, head 1e-3).
- Write-up: [docs/baseline-submission.md](docs/baseline-submission.md)

### 2. Improved text pipeline

A review of the baseline ([docs/plan.md](docs/plan.md)) showed the model was being asked to read numbers like pack size and volume as plain tokens. The improved pipeline ([docs/what-was-done.md](docs/what-was-done.md)):

- **Feature engineering** — parses `catalog_content` into `value`, `unit` (one-hot) and `pack_count` (standardised), plus a `clean_text` field for the transformer.
- **Multimodal head** — concatenates the DeBERTa embedding with the numerical features before the regression head.
- **Mean pooling** — optional alternative to the CLS token.
- **5-fold cross-validation** — trains five models and averages their test predictions.

Code: [`src/main_improved.py`](src/main_improved.py), [`src/data_utils.py`](src/data_utils.py), [`src/deberta_price_predictor.py`](src/deberta_price_predictor.py)

### 3. Image model: CLIP + LoRA

An independent image-based predictor ([docs/image-plan.md](docs/image-plan.md)) that can be ensembled with the text model:

- Frozen `openai/clip-vit-large-patch14` vision encoder with LoRA adapters (rank 16) on the attention `q_proj`/`v_proj` layers, plus a regression head.
- Images streamed directly from S3; bf16 and `torch.compile` for H100 throughput.

Code and usage: [`image_model/`](image_model/README.md)

## Results

| Model | Validation SMAPE | Status |
|---|---|---|
| Baseline DeBERTa-v3-large (single split) | **49.23%** | Measured, submitted 13 Oct 2025 |
| Improved text pipeline (5-fold ensemble) | 32–39% | Target from the improvement plan; not yet recorded |
| CLIP + LoRA image model | 25–35% | Target; not yet recorded |

Running either pipeline writes its actual metrics to the output folder (`kfold_results.json` for the text model).

## How to Run

Requirements: Python 3.12+, an NVIDIA GPU (the defaults are tuned for an H100 80GB), and [uv](https://docs.astral.sh/uv/) (the setup script installs it if missing).

### Text model

```bash
# Full pipeline: setup, feature extraction, 5-fold training, ensemble submission
./scripts/setup-and-run.sh

# Quick check: single split, no k-fold (~30–60 min)
./scripts/setup-and-run.sh --quick

# Other modes
./scripts/setup-and-run.sh --setup-only   # install environment only
./scripts/setup-and-run.sh --status       # show environment / GPU status
```

The full 5-fold run takes roughly 2.5–5 hours on a single H100.

Manual run:

```bash
source .venv/bin/activate
python3 src/main_improved.py                      # 5-fold CV + ensemble
python3 src/main_improved.py --no-kfold           # single split
python3 src/main_improved.py --use-mean-pooling   # mean pooling instead of CLS
python3 src/main_improved.py --model-name microsoft/deberta-v3-base   # smaller model
```

Re-generate a submission from saved fold models:

```bash
./scripts/predict.sh output/improved/run_TIMESTAMP [n_folds]
```

Output:

```
output/improved/run_TIMESTAMP/
├── submission.csv        # predictions for test.csv
├── kfold_results.json    # per-fold and mean ± std SMAPE
├── config.json           # training configuration
└── fold_1/ … fold_5/     # fold models
```

### Image model

Needs AWS credentials with read access to the image bucket configured in [`image_model/config.py`](image_model/config.py).

```bash
./image_model/setup.sh
python image_model/verify_setup.py      # optional
./image_model/run_training.sh           # or --quick
```

Predictions are written to `image_model/output/submission_*.csv`.

### Utilities

```bash
python scripts/check_data_preprocessing.py   # sanity-check feature extraction on the data
./scripts/fix_permissions.sh                 # fix write permissions on output/
```

## Project Structure

```
├── src/                          # Text model
│   ├── main_improved.py          # Improved pipeline (features + k-fold ensemble)
│   ├── main.py                   # Original baseline pipeline
│   ├── predict_ensemble.py       # Predict with saved fold models
│   ├── data_utils.py             # Cleaning + catalog feature extraction
│   ├── deberta_price_predictor.py# Dataset, model and trainer
│   ├── evaluation.py             # SMAPE and per-price-range metrics
│   └── config.py                 # Hyperparameters
├── image_model/                  # CLIP + LoRA image model
├── scripts/                      # Setup, run, predict and utility scripts
├── docs/                         # Development process
│   ├── baseline-submission.md    # Baseline write-up and score
│   ├── plan.md                   # Text model improvement plan
│   ├── what-was-done.md          # What the improved pipeline changed
│   └── image-plan.md             # Image model plan
├── data/                         # train.csv / test.csv (not tracked)
└── pyproject.toml
```
