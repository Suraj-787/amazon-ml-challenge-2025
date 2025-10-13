# Image Model - Setup Complete ✅

## What Was Created

A complete image-based price prediction system in the `image_model/` folder.

## Structure

```
image_model/
├── config.py              # Configuration
├── model.py               # CLIP + LoRA model
├── trainer.py             # Training loop
├── s3_dataset.py          # S3 streaming dataset
├── main.py                # Main script
├── requirements.txt       # Dependencies
├── setup.sh              # Setup script
├── run_training.sh       # Training script ⭐
├── verify_setup.py       # Verification script
└── README.md             # Documentation
```

## How to Use

```bash
# Setup
./image_model/setup.sh

# Verify (optional)
python image_model/verify_setup.py

# Train
./image_model/run_training.sh

# Or quick test
./image_model/run_training.sh --quick
```

## Key Features

- ✅ LoRA fine-tuning (only train 2% of params)
- ✅ S3 streaming (35k images from `s3://amzn-ml/images/train/`)
- ✅ H100 optimized (batch_size=512, bf16, torch.compile)
- ✅ Early stopping
- ✅ Automatic submission generation

## Expected Results

- **Training Time**: 2-4 hours on H100
- **SMAPE**: 25-35%
- **Output**: `image_model/output/submission_*.csv`

## Configuration

Default settings in `config.py`:
- Base model: `openai/clip-vit-large-patch14`
- LoRA rank: 16
- Batch size: 512 (H100)
- Epochs: 10

Adjust via command line:
```bash
./image_model/run_training.sh --batch-size 256 --lora-r 32 --epochs 15
```

## Separate from Text Model

This is independent from the DeBERTa text model in `src/`. Both can be trained separately and ensembled.

```
ml-challenge-2025/
├── src/              # Text model (DeBERTa)
├── image_model/      # Image model (CLIP + LoRA) ⭐ NEW
└── data/             # Shared dataset
```

## Next Steps

1. Run `./image_model/setup.sh`
2. Run `./image_model/run_training.sh`
3. Submit `image_model/output/submission_*.csv`

---

**Ready to train!** See `image_model/README.md` for details.
