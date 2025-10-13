# ML Challenge 2025 - Team Monk Builders

**Team Members:** Suraj Kumar, Himanshu Rai, Pratik Gautam, Impana KS

## 🎯 Improved DeBERTa Price Prediction

This solution implements **all key improvements from plan.md** for significantly better performance:

1. ✅ **Feature Engineering** - Extract value, unit, pack_count from text
2. ✅ **Multimodal Architecture** - Combine text + numerical features
3. ✅ **K-Fold Cross-Validation** - 5-fold CV with ensemble
4. ✅ **Mean Pooling** - Optional alternative to CLS token

**Expected Improvement:** 18-33% SMAPE reduction (from ~48% to 32-39%)

## 🚀 One Command to Run Everything

```bash
./scripts/setup-and-run.sh
```

This will:
1. ✅ Check prerequisites and install dependencies
2. ✅ Extract structured features from catalog_content
3. ✅ Train 5 models with k-fold cross-validation
4. ✅ Create ensemble predictions
5. ✅ Save submission.csv

**Expected Time:** 1-2 hours on H100 80GB GPU

## ⚡ Quick Test (Fast)

```bash
./scripts/setup-and-run.sh --quick
```

Runs in ~30-60 minutes without k-fold.

## 📊 What Was Improved

### 1. Feature Engineering
**Before:** Model sees "Value: 72.0 Unit: Fl Oz" as text  
**After:** Model gets explicit features: value=72.0, unit=fl_oz, pack_count=6

### 2. Multimodal Architecture
**Before:** Only text embeddings → regression head  
**After:** Text embeddings + numerical features → regression head

### 3. K-Fold Cross-Validation
**Before:** Single model on one train/val split  
**After:** 5 models on different splits, ensemble predictions

### 4. Mean Pooling
**Before:** Only CLS token  
**After:** Optional mean pooling over all tokens

## 📁 Project Structure

```
ml-challenge-2025/
├── src/
│   ├── main_improved.py           # 🆕 Improved pipeline with k-fold CV
│   ├── data_utils.py              # ✨ Updated with feature extraction
│   ├── deberta_price_predictor.py # ✨ Updated with multimodal support
│   ├── config.py                  # Model configuration
│   └── evaluation.py              # Metrics and evaluation
├── scripts/
│   ├── quick-start.sh             # Quick setup
│   └── setup-and-run.sh           # ✨ Updated for improved pipeline
├── data/
│   ├── train.csv                  # Training data
│   └── test.csv                   # Test data
├── output/
│   └── improved/                  # 🆕 Improved model outputs
│       └── run_TIMESTAMP/
│           ├── submission.csv     # Final predictions
│           ├── kfold_results.json # K-fold metrics
│           └── fold_1/ ... fold_5/ # Individual models
└── pyproject.toml                 # Dependencies
```

## 🎯 Model Architecture (H100 Optimized)

- **Base Model:** microsoft/deberta-v3-large (304M parameters)
- **Batch Size:** 64 (optimized for H100 80GB)
- **Training:** 5-fold CV × 3 epochs = 15 epochs total
- **Feature Engineering:** Extract value, unit, pack_count
- **Multimodal Input:** Text embeddings (1024 dim) + numerical features
- **Pooling:** CLS token or mean pooling (configurable)
- **Regression Head:** 3-layer MLP with GELU activation
- **Training:** 5-fold CV with ensemble predictions

## 📈 Expected Performance

| Metric | Before | After (Expected) | Improvement |
|--------|--------|------------------|-------------|
| SMAPE | ~48% | 28-36% | 25-40% reduction |
| Training | Single model | 5-model ensemble | More robust |
| Features | Text only | Text + numerical | Better predictions |
| Model | Base (86M) | Large (304M) | Better accuracy |
| Hardware | Generic | H100 optimized | 2-4x faster |

## 💻 Manual Run (if needed)

```bash
# Setup environment
./scripts/setup-and-run.sh --setup-only

# Activate environment
source .venv/bin/activate

# Run improved pipeline
python3 src/main_improved.py

# Or without k-fold (faster)
python3 src/main_improved.py --no-kfold

# With mean pooling
python3 src/main_improved.py --use-mean-pooling
```

## 🔍 Check Status

```bash
./scripts/setup-and-run.sh --status
```

## 📊 Output

After training, find your results in:
```
output/improved/run_TIMESTAMP/
├── submission.csv          # Submit this!
├── kfold_results.json     # CV metrics
├── config.json            # Training config
└── fold_1/ ... fold_5/    # Individual models
```

## 🎓 Key Improvements Explained

### Feature Extraction
Parses catalog_content to extract:
- `value`: Product size/amount (e.g., 11.4)
- `unit`: Measurement unit (e.g., "fl oz") - one-hot encoded
- `pack_count`: Number of items (e.g., 6 from "Pack of 6")
- `clean_text`: Structured text for DeBERTa

### Multimodal Architecture
Combines semantic understanding (text) with explicit numerical features for better predictions.

### K-Fold Ensemble
Trains 5 models on different data splits and averages predictions for more robust results.

## 📚 Documentation

- **RUN_INSTRUCTIONS.md** - Detailed run instructions
- **WHAT_WAS_DONE.md** - Summary of improvements
- **plan.md** - Original improvement recommendations

## 🤝 Team Contributions

- **Suraj Kumar**: Model architecture and improvements
- **Himanshu Rai**: Feature engineering and data preprocessing
- **Pratik Gautam**: K-fold CV and evaluation
- **Impana KS**: Documentation and testing

---

**Team Monk Builders** | ML Challenge 2025 | Improved Pipeline 🚀
