# Run Instructions - EC2 P5

## One Command to Run Everything

```bash
./scripts/setup-and-run.sh
```

That's it! This will:
1. ✅ Check prerequisites (Python, GPU, etc.)
2. ✅ Install all dependencies
3. ✅ Extract features from catalog_content (value, unit, pack_count)
4. ✅ Train 5 models with k-fold cross-validation
5. ✅ Create ensemble predictions
6. ✅ Save submission.csv

## Expected Results

- **Training Time:** ~2.5-5 hours on P5 GPU
- **Expected SMAPE:** 32-39% (down from ~48% baseline)
- **Output:** `output/improved/run_TIMESTAMP/submission.csv`

## Quick Test (Fast)

If you want to test first without k-fold:

```bash
./scripts/setup-and-run.sh --quick
```

This runs in ~30-60 minutes.

## What Was Improved

1. **Feature Engineering** - Extracts value, unit, pack_count from text
2. **Multimodal Model** - Combines text + numerical features
3. **K-Fold CV** - 5-fold cross-validation with ensemble
4. **Mean Pooling** - Optional (can enable with flag)

## Manual Run (if needed)

```bash
# Setup environment
./scripts/setup-and-run.sh --setup-only

# Activate environment
source .venv/bin/activate

# Run training
python3 src/main_improved.py

# Or without k-fold (faster)
python3 src/main_improved.py --no-kfold
```

## Check Status

```bash
./scripts/setup-and-run.sh --status
```

## Help

```bash
./scripts/setup-and-run.sh --help
```

## Output Location

After training completes, find your submission file at:
```
output/improved/run_TIMESTAMP/submission.csv
```

Submit this file to the competition!
