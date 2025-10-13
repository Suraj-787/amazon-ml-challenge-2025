#!/bin/bash
#
# Setup script for image model training
#

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up Image Model Environment${NC}"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo -e "${GREEN}Installing PyTorch with CUDA support...${NC}"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo -e "${GREEN}Installing other dependencies...${NC}"
pip install -r image_model/requirements.txt

# Verify installation
echo ""
echo -e "${GREEN}Verifying installation...${NC}"
python3 << EOF
import torch
import transformers
import peft
import boto3

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ PEFT: {peft.__version__}")
print(f"✓ Boto3: {boto3.__version__}")
EOF

# Create output directory
mkdir -p image_model/output

# Check AWS credentials
echo ""
echo -e "${GREEN}Checking AWS credentials...${NC}"
if [ -f ".env" ]; then
    echo "✓ .env file found"
else
    echo -e "${YELLOW}⚠ .env file not found. Make sure AWS credentials are configured.${NC}"
fi

# Test S3 access
python3 << EOF
import boto3
try:
    s3 = boto3.client('s3', region_name='us-east-1')
    s3.head_bucket(Bucket='amzn-ml')
    print('✓ S3 bucket access confirmed')
except Exception as e:
    print(f'✗ S3 access failed: {e}')
    print('  Make sure AWS credentials are configured')
EOF

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run training: ./image_model/run_training.sh"
echo "  3. Or quick test: ./image_model/run_training.sh --quick"
echo ""
