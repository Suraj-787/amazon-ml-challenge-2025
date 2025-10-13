# Image Model - LoRA Fine-tuning for Price Prediction

Image-based price prediction using CLIP + LoRA, optimized for H100 GPU.

## Quick Start

```bash
# 1. Setup
./image_model/setup.sh
source .venv/bin/activate

# 2. Verify (optional)
python image_model/verify_setup.py

# 3. Train
./image_model/run_training.sh

# Or quick test
./image_model/run_training.sh --quick
```

## Architecture

```
Image (224x224) 
  → CLIP Encoder (frozen, 304M params)
  → LoRA Adapters (trainable, ~2M params)
  → Regression Head (trainable, ~1M params)
  → Price Prediction
```

## Dataset

- **Training**: 35,000 images in `s3://amzn-ml/images/train/`
- **Format**: `{sample_id}.jpg` (or .png, .webp)
- **Streaming**: Direct from S3, no disk storage needed

## Configuration

Edit `config.py` or use command line:

```bash
./image_model/run_training.sh --batch-size 256 --epochs 15 --lora-r 32
```

Key settings:
- `batch_size`: 512 (H100), 256 (A100), 128 (smaller GPUs)
- `lora_r`: 8 (fast), 16 (balanced), 32 (best)
- `num_epochs`: 10 (default)

## Output

```
image_model/output/
├── best_model/              # Model checkpoint (~15MB)
├── submission_*.csv         # Predictions (submit this!)
└── training.log             # Logs
```

## Expected Performance

- **Training Time**: 2-4 hours on H100
- **SMAPE**: 25-35%
- **Model Size**: ~15MB

## Troubleshooting

**Out of Memory?**
```bash
./image_model/run_training.sh --batch-size 128
```

**S3 Access Issues?**
```bash
aws s3 ls s3://amzn-ml/images/train/
```

**Check Setup:**
```bash
python image_model/verify_setup.py
```

## Files

- `config.py` - Configuration
- `model.py` - CLIP + LoRA architecture
- `trainer.py` - Training loop
- `s3_dataset.py` - S3 streaming dataset
- `main.py` - Main script
- `run_training.sh` - Training script
- `setup.sh` - Environment setup
- `verify_setup.py` - Setup verification
