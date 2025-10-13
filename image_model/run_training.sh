#!/bin/bash
#
# Image-based Price Prediction Training Script
# Optimized for H100 80GB GPU
#
# Usage:
#   ./image_model/run_training.sh              # Full training
#   ./image_model/run_training.sh --quick      # Quick test
#   ./image_model/run_training.sh --help       # Show help
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# Check if running on GPU
check_gpu() {
    print_msg "Checking GPU availability..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. GPU required for training!"
        exit 1
    fi
    
    # Check GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    
    print_msg "GPU: $GPU_NAME"
    print_msg "GPU Memory: ${GPU_MEM} MB"
    
    if [ "$GPU_MEM" -lt 40000 ]; then
        print_warning "GPU has less than 40GB memory. Consider reducing batch size."
    fi
}

# Check Python environment
check_python() {
    print_msg "Checking Python environment..."
    
    if [ ! -d ".venv" ]; then
        print_error "Virtual environment not found. Please run setup first:"
        echo "  python -m venv .venv"
        echo "  source .venv/bin/activate"
        echo "  pip install -r image_model/requirements.txt"
        exit 1
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    print_msg "Python: $(python --version)"
    print_msg "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
    print_msg "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
}

# Check AWS credentials
check_aws() {
    print_msg "Checking AWS credentials..."
    
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Make sure AWS credentials are configured."
    fi
    
    # Test S3 access
    python -c "
import boto3
try:
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.head_bucket(Bucket='amzn-ml')
    print('✓ S3 access confirmed')
except Exception as e:
    print(f'✗ S3 access failed: {e}')
    exit(1)
" || {
        print_error "Cannot access S3 bucket. Check AWS credentials."
        exit 1
    }
}

# Check data files
check_data() {
    print_msg "Checking data files..."
    
    if [ ! -f "data/train.csv" ]; then
        print_error "data/train.csv not found!"
        exit 1
    fi
    
    if [ ! -f "data/test.csv" ]; then
        print_error "data/test.csv not found!"
        exit 1
    fi
    
    TRAIN_ROWS=$(wc -l < data/train.csv)
    TEST_ROWS=$(wc -l < data/test.csv)
    
    print_msg "Train samples: $TRAIN_ROWS"
    print_msg "Test samples: $TEST_ROWS"
}

# Main training function
run_training() {
    print_header "Starting Image-based Price Prediction Training"
    
    # Pre-flight checks
    check_gpu
    check_python
    check_aws
    check_data
    
    # Create output directory
    mkdir -p image_model/output
    
    print_header "Training Configuration"
    
    # Parse arguments
    ARGS="$@"
    
    if [[ "$ARGS" == *"--quick"* ]]; then
        print_msg "Mode: Quick test (reduced settings)"
    else
        print_msg "Mode: Full training (optimized for H100)"
    fi
    
    # Run training
    print_header "Starting Training Loop"
    
    python image_model/main.py $ARGS
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        print_header "Training Completed Successfully!"
        print_msg "Check image_model/output/ for results"
        print_msg "Submission file: image_model/output/submission_*.csv"
    else
        print_error "Training failed with exit code $EXIT_CODE"
        exit $EXIT_CODE
    fi
}

# Show help
show_help() {
    cat << EOF
Image-based Price Prediction Training Script

Usage:
  ./image_model/run_training.sh [OPTIONS]

Options:
  --quick              Quick test run with reduced settings
  --predict-only       Only make predictions (skip training)
  --checkpoint DIR     Use specific checkpoint for prediction
  --batch-size N       Batch size (default: 512 for H100)
  --epochs N           Number of epochs (default: 10)
  --lora-r N           LoRA rank (default: 16)
  --train-limit N      Limit training samples (default: 35000)
  --help               Show this help message

Examples:
  # Full training on H100
  ./image_model/run_training.sh

  # Quick test run
  ./image_model/run_training.sh --quick

  # Custom batch size
  ./image_model/run_training.sh --batch-size 256

  # Only predictions
  ./image_model/run_training.sh --predict-only --checkpoint image_model/output/best_model

Environment:
  Requires:
  - CUDA-capable GPU (H100 recommended)
  - Python virtual environment (.venv)
  - AWS credentials configured
  - S3 bucket 'amzn-ml' accessible

Output:
  - Training logs: image_model/training.log
  - Model weights: image_model/output/best_model/
  - Submission: image_model/output/submission_*.csv
  - Training history: image_model/output/training_history.json

EOF
}

# Main entry point
main() {
    if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    run_training "$@"
}

main "$@"
