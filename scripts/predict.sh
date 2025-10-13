#!/bin/bash

# Ensemble Prediction Script
# Generates submission.csv using trained fold models

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║           Ensemble Prediction - ML Challenge 2025       ║"
echo "║                    Team: Monk Builders                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if model directory is provided
if [ -z "$1" ]; then
    echo "Usage: ./scripts/predict.sh <model_directory>"
    echo "Example: ./scripts/predict.sh output/improved/run_20251013_145231"
    exit 1
fi

MODEL_DIR="$1"
N_FOLDS="${2:-3}"  # Default to 3 folds

echo "📁 Model Directory: $MODEL_DIR"
echo "🔢 Using $N_FOLDS folds for ensemble"
echo ""

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import torch; import transformers; import pandas" 2>/dev/null || {
    echo "❌ Error: Required packages not installed. Please run setup first."
    exit 1
}

echo "✅ Dependencies verified"
echo ""

# Run prediction
echo "🚀 Starting ensemble prediction..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python src/predict_ensemble.py \
    --model-dir "$MODEL_DIR" \
    --n-folds "$N_FOLDS" \
    --data-dir data \
    --model-name microsoft/deberta-v3-large

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS! Submission file created:"
    echo "   📄 $MODEL_DIR/submission.csv"
    echo ""
    echo "Next steps:"
    echo "1. Download the submission file from EC2"
    echo "2. Submit to the competition platform"
    echo ""
else
    echo ""
    echo "❌ Prediction failed with exit code: $EXIT_CODE"
    echo "Please check the error messages above."
    echo ""
    exit $EXIT_CODE
fi
