# What Was Done

Development log for the improved text pipeline. Starting point: the baseline DeBERTa-v3-large model described in [baseline-submission.md](baseline-submission.md) (49.23% validation SMAPE). The four priority improvements from [plan.md](plan.md) were implemented.

## Improvements Implemented

### 1. Feature Engineering ✅
- Extracts `value`, `unit`, `pack_count` from `catalog_content`
- Creates `clean_text` for DeBERTa (item name + bullets + description)
- Standardizes numerical features with `StandardScaler`
- One-hot encodes the categorical `unit` feature

### 2. Multimodal Model Architecture ✅
- The model accepts both text embeddings and numerical features
- They are concatenated before the regression head
- The head's input size adjusts to the number of features

### 3. Mean Pooling Option ✅
- Added a `use_mean_pooling` parameter
- Mean pooling can be used instead of the CLS token
- Configurable from the command line (`--use-mean-pooling`)

### 4. K-Fold Cross-Validation ✅
- Trains 5 separate models on different data splits
- Ensembles predictions by averaging all 5 models
- Reports mean ± std SMAPE across folds

## Files Modified

1. **src/data_utils.py** — added feature extraction
2. **src/deberta_price_predictor.py** — added multimodal support
3. **src/main_improved.py** — new main script with k-fold CV
4. **scripts/setup-and-run.sh** — updated to use the improved pipeline

## Before vs After

| | Before (baseline) | After (improved) |
|---|---|---|
| Input | Text only | Text + numerical features |
| Models | Single model | 5-model k-fold ensemble |
| Pooling | CLS token only | CLS or mean pooling |
| Validation SMAPE | 49.23% (measured) | 32–39% (target from the plan) |

## Run Time Estimates (EC2 P5 / H100)

| Command | What it runs | Estimated time |
|---|---|---|
| `./scripts/setup-and-run.sh --quick` | Single split, no k-fold | 30–60 min |
| `./scripts/setup-and-run.sh` | Full 5-fold CV + ensemble | 2.5–5 hours |

## What Happens When You Run It

1. Checks Python, GPU and dependencies
2. Creates a virtual environment
3. Installs PyTorch, Transformers, etc.
4. Loads `train.csv` and `test.csv`
5. Extracts features (`value`, `unit`, `pack_count`)
6. Trains 5 models with k-fold CV
7. Makes ensemble predictions
8. Saves `submission.csv` to `output/improved/run_TIMESTAMP/`
