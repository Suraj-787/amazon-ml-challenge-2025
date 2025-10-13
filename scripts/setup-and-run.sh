#!/bin/bash

# ML Challenge 2025 - DeBERTa-V3-Large Setup and Run Script
# Team: Monk Builders
# Members: Suraj Kumar, Himanshu Rai, Pratik Gautam, Impana KS

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Print banner
print_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                 ML Challenge 2025                        ║"
    echo "║      Improved DeBERTa Price Prediction Pipeline         ║"
    echo "║                  Team: Monk Builders                     ║"
    echo "║      Suraj Kumar, Himanshu Rai, Pratik Gautam, Impana KS║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "${CYAN}🚀 Enhanced with Feature Engineering + Multimodal Architecture${NC}"
    echo -e "${CYAN}   Expected 18-33% SMAPE improvement with K-Fold CV${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Check if running from project root
check_location() {
    if [[ ! -f "pyproject.toml" ]] || [[ ! -d "src" ]]; then
        print_error "Please run this script from the ml-challenge-2025 project root directory"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    print_step "1" "Checking prerequisites for DeBERTa pipeline..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or newer."
        exit 1
    fi

    python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    print_success "Python $python_version detected"

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        print_step "1.1" "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"

        # Reload shell environment
        source ~/.bashrc 2>/dev/null || source ~/.profile 2>/dev/null || true

        if ! command -v uv &> /dev/null; then
            print_error "Failed to install uv. Please install manually or restart terminal."
            print_error "You can manually install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
        print_success "uv installed successfully"
    else
        print_success "uv is already available"
    fi

    # Check GPU availability with detailed info for DeBERTa
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "Unknown GPU")
        gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

        print_success "NVIDIA GPU detected: $gpu_info"

        if [[ $gpu_memory -ge 8000 ]]; then
            print_success "GPU memory: ${gpu_memory}MB - Perfect for DeBERTa-V3-Large!"
            print_info "Expected training time: 2-3 hours for full dataset"
        elif [[ $gpu_memory -ge 4000 ]]; then
            print_warning "GPU memory: ${gpu_memory}MB - May need reduced batch size"
            print_info "Will automatically adjust batch size for your GPU"
        else
            print_warning "GPU memory: ${gpu_memory}MB - Training will be slow"
            print_info "Consider using --quick-test mode for initial testing"
        fi
    else
        print_warning "No GPU detected - will use CPU mode"
        print_warning "Note: DeBERTa training will be very slow on CPU (hours to days)"
        print_info "Strongly recommend using GPU for DeBERTa-V3-Large training"
    fi

    # Check internet connectivity for DeBERTa model download
    if ping -c 1 huggingface.co &> /dev/null; then
        print_success "Internet connectivity verified (needed for DeBERTa model download)"
        print_info "Will download microsoft/deberta-v3-large (~1.2GB) on first run"
    else
        print_warning "Cannot reach huggingface.co - model download may fail"
    fi

    # Check disk space (need ~4GB for DeBERTa model + outputs)
    available_space=$(df . | awk 'NR==2 {print $4}')
    if [[ $available_space -gt 4000000 ]]; then
        print_success "Sufficient disk space available"
    else
        print_warning "Low disk space - need ~4GB for DeBERTa model and outputs"
    fi
}

# Check data files
check_data_files() {
    print_step "2" "Checking data files..."

    if [[ ! -f "data/train.csv" ]]; then
        print_error "data/train.csv not found!"
        print_error "Please ensure your data files are in the data/ directory"
        exit 1
    fi

    if [[ ! -f "data/test.csv" ]]; then
        print_error "data/test.csv not found!"
        print_error "Please ensure your data files are in the data/ directory"
        exit 1
    fi

    # Check data file sizes and content
    train_lines=$(wc -l < data/train.csv)
    test_lines=$(wc -l < data/test.csv)

    print_success "Data files found:"
    print_success "  Train: $((train_lines - 1)) samples"
    print_success "  Test: $((test_lines - 1)) samples"

    # Quick validation of required columns
    if head -1 data/train.csv | grep -q "catalog_content"; then
        print_success "Required columns detected in training data"
    else
        print_warning "catalog_content column not found - please verify data format"
    fi

    if head -1 data/train.csv | grep -q "price"; then
        print_success "Price column detected in training data"
    else
        print_error "price column not found in training data!"
        exit 1
    fi

    # Note: Detailed data verification will be run after dependency installation
    print_info "Basic data validation passed - detailed verification after dependency setup"
}

# Setup directories
setup_directories() {
    print_step "3" "Setting up directories..."

    # Fix ownership of existing output directory if needed
    if [[ -d "output" ]]; then
        sudo chown -R $(whoami):$(whoami) output/ 2>/dev/null || true
        sudo chmod -R 755 output/ 2>/dev/null || true
    fi

    # Create all necessary directories
    mkdir -p output/improved notebooks models

    # Fix permissions for output directories
    chmod -R 755 output notebooks models 2>/dev/null || true

    # Ensure current user owns all directories
    chown -R $(whoami):$(whoami) output/ notebooks/ models/ 2>/dev/null || true

    # Verify we can write to output directory
    if touch output/.test_write 2>/dev/null; then
        rm output/.test_write
        print_success "Directories created with proper permissions"
    else
        print_warning "Cannot write to output/ directory, attempting to fix..."
        sudo chown -R $(whoami):$(whoami) output/ 2>/dev/null || true
        sudo chmod -R 755 output/ 2>/dev/null || true
        
        if touch output/.test_write 2>/dev/null; then
            rm output/.test_write
            print_success "Permissions fixed successfully"
        else
            print_error "Cannot write to output/ directory. Please run:"
            print_error "  sudo chown -R \$(whoami):\$(whoami) output/"
            print_error "  sudo chmod -R 755 output/"
            exit 1
        fi
    fi
    
    print_success "  Output directory: output/ (for models and predictions)"
    print_success "  Models directory: models/ (for saved checkpoints)"
}

# Setup virtual environment and install dependencies
setup_environment() {
    print_step "4" "Setting up Python environment for DeBERTa..."

    # Create virtual environment with uv
    if [[ ! -d ".venv" ]]; then
        print_step "4.1" "Creating virtual environment..."
        uv venv .venv --python python3
        print_success "Virtual environment created"
    else
        print_success "Virtual environment already exists"
    fi

    # Activate virtual environment
    source .venv/bin/activate

    # Install core dependencies first
    print_step "4.2" "Installing core ML dependencies..."
    uv pip install \
        pandas>=2.0.0 \
        numpy>=1.24.0 \
        scikit-learn>=1.3.0 \
        scipy>=1.10.0 \
        tqdm>=4.64.0 \
        protobuf>=3.20.0 \
        sentencepiece>=0.1.97

    print_success "Core dependencies installed"

    # Install PyTorch with appropriate CUDA support
    print_step "4.3" "Installing PyTorch for DeBERTa..."
    if command -v nvidia-smi &> /dev/null; then
        print_info "Installing PyTorch with CUDA support for GPU acceleration..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        print_success "PyTorch with CUDA installed"
    else
        print_info "Installing PyTorch CPU version..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        print_success "PyTorch CPU installed"
    fi

    # Install Transformers and related dependencies
    print_step "4.4" "Installing Transformers library for DeBERTa..."
    uv pip install \
        transformers>=4.21.0 \
        tokenizers>=0.13.0 \
        accelerate>=0.20.0

    print_success "Transformers library installed"

    # Install additional utilities
    print_step "4.5" "Installing additional utilities..."
    uv pip install \
        matplotlib>=3.7.0 \
        seaborn>=0.12.0

    print_success "Additional utilities installed"

    print_success "✅ All dependencies installed successfully!"
    print_info "Environment ready for DeBERTa-V3-Large training"
}

# Test DeBERTa model availability
test_deberta_model() {
    print_step "5" "Testing DeBERTa model availability..."

    # Activate virtual environment
    source .venv/bin/activate

    # Test if we can import required packages
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"

    # Test transformers import
    python3 -c "
import transformers
print(f'Transformers version: {transformers.__version__}')
"

    # Test DeBERTa tokenizer download (small test)
    print_info "Testing DeBERTa tokenizer download..."
    python3 -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
    print('✅ DeBERTa tokenizer loaded successfully')
except Exception as e:
    print(f'❌ Error loading DeBERTa tokenizer: {e}')
    exit(1)
"

    print_success "DeBERTa model components verified"
    print_info "Full model (~1.2GB) will be downloaded on first training run"
}

# Run detailed data verification after dependencies are installed
run_detailed_data_verification() {
    print_step "5.5" "Running detailed data verification..."

    # Activate virtual environment
    source .venv/bin/activate

    # Run detailed data verification if available
    if [[ -f "check_data.py" ]]; then
        print_info "Running comprehensive data verification with pandas..."
        if python3 check_data.py; then
            print_success "Detailed data verification passed"
        else
            print_error "Data verification failed - please check your data files"
            exit 1
        fi
    else
        print_info "check_data.py not found - skipping detailed verification"
    fi
}

# Run the improved DeBERTa ML pipeline
run_pipeline() {
    print_step "6" "Running Improved DeBERTa ML Pipeline with Feature Engineering..."

    # Activate virtual environment
    source .venv/bin/activate

    print_info "Pipeline Overview (H100 Optimized):"
    print_info "  • Model: microsoft/deberta-v3-large (304M parameters)"
    print_info "  • Batch Size: 64 (optimized for H100 80GB)"
    print_info "  • Training: 5-fold CV × 3 epochs = 15 epochs total"
    print_info "  • Feature Engineering: Extract value, unit, pack_count from text"
    print_info "  • Multimodal: Text embeddings + numerical features"
    print_info "  • K-Fold CV: 5-fold cross-validation with ensemble"
    print_info "  • Expected improvement: 25-40% SMAPE reduction"
    echo ""

    # Check if this is a quick test or full run
    MODE=${1:-full}

    case $MODE in
        "quick"|"test")
            print_info "🚀 Running QUICK TEST mode (no k-fold, fast verification)"
            print_info "   Purpose: Verify everything works before full training"
            echo ""

            if python3 src/main_improved.py --no-kfold; then
                print_success "🎉 Quick test completed successfully!"
                print_info "✅ Your environment is ready for full training"
                echo ""
                print_info "Next steps:"
                print_info "  • Run full training: $0 --full"
                print_info "  • Or use: python3 src/main_improved.py"
            else
                print_error "❌ Quick test failed - please check the error messages above"
                exit 1
            fi
            ;;

        "full"|*)
            print_info "🚀 Running FULL TRAINING mode (5-fold CV, H100 optimized)"
            print_info "   Expected time: 1-2 hours on H100 80GB"
            echo ""

            # Run the improved pipeline with all features
            if python3 src/main_improved.py; then
                print_success "🎉 Improved DeBERTa pipeline completed successfully!"
                echo ""
                print_success "📊 Results Summary:"

                # Find the latest output directory
                latest_output=$(find output/improved -type d -name "run_*" | sort | tail -1)
                if [[ -n "$latest_output" ]]; then
                    print_success "  📁 Output directory: $latest_output"

                    if [[ -f "$latest_output/submission.csv" ]]; then
                        num_predictions=$(tail -n +2 "$latest_output/submission.csv" | wc -l)
                        print_success "  📋 Submission file: $latest_output/submission.csv ($num_predictions predictions)"
                    fi

                    if [[ -f "$latest_output/kfold_results.json" ]]; then
                        print_success "  📈 K-Fold results: $latest_output/kfold_results.json"

                        # Try to extract SMAPE score
                        if command -v jq &> /dev/null; then
                            avg_smape=$(jq -r '.average_smape // "N/A"' "$latest_output/kfold_results.json" 2>/dev/null)
                            std_smape=$(jq -r '.std_smape // "N/A"' "$latest_output/kfold_results.json" 2>/dev/null)
                            if [[ "$avg_smape" != "N/A" ]]; then
                                print_success "  🎯 Average SMAPE: ${avg_smape}% ± ${std_smape}%"
                            fi
                        fi
                    fi

                    # Count fold models
                    fold_count=$(find "$latest_output" -type d -name "fold_*" | wc -l)
                    if [[ $fold_count -gt 0 ]]; then
                        print_success "  🏆 Trained $fold_count models (ensemble)"
                    fi
                else
                    print_success "  📁 Check the output/improved/ directory for results"
                fi

                echo ""
                print_success "🏆 Ready for submission! Use the generated submission.csv file"

            else
                print_error "❌ Pipeline failed - please check the error messages above"
                exit 1
            fi
            ;;
    esac
}

# Display help
show_help() {
    echo -e "${YELLOW}Improved DeBERTa Price Prediction Pipeline${NC}"
    echo ""
    echo -e "${BLUE}USAGE:${NC}"
    echo "  $0 [MODE]"
    echo ""
    echo -e "${BLUE}MODES:${NC}"
    echo "  --setup-only    Setup environment and dependencies only"
    echo "  --quick         Run quick test (no k-fold, fast verification)"
    echo "  --full          Run full training pipeline with k-fold CV (default)"
    echo "  --status        Show current environment status"
    echo "  --help          Show this help message"
    echo ""
    echo -e "${BLUE}EXAMPLES:${NC}"
    echo "  $0                              # Full setup + training with k-fold CV"
    echo "  $0 --setup-only                 # Setup environment only"
    echo "  $0 --quick                      # Quick test run (no k-fold)"
    echo "  $0 --status                     # Check environment status"
    echo ""
    echo -e "${BLUE}FEATURES:${NC}"
    echo "  • Feature Engineering: Extract value, unit, pack_count from text"
    echo "  • Multimodal Architecture: Text embeddings + numerical features"
    echo "  • K-Fold Cross-Validation: 5-fold CV with ensemble predictions"
    echo "  • Mean Pooling: Optional alternative to CLS token"
    echo "  • DeBERTa-V3-Base: Fast and efficient (86M parameters)"
    echo "  • Expected: 18-33% SMAPE reduction from baseline"
    echo ""
}

# Show current status
show_status() {
    print_step "STATUS" "Current environment status..."

    # Check if virtual environment exists
    if [[ -d ".venv" ]]; then
        print_success "Virtual environment: Available"

        # Check if activated
        if [[ "$VIRTUAL_ENV" == *".venv"* ]]; then
            print_success "Virtual environment: Activated"
        else
            print_info "Virtual environment: Not activated (will be activated automatically)"
        fi
    else
        print_warning "Virtual environment: Not created"
    fi

    # Check key dependencies
    if [[ -d ".venv" ]]; then
        source .venv/bin/activate

        if python3 -c "import torch" 2>/dev/null; then
            torch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
            print_success "PyTorch: $torch_version"

            if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
                gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
                print_success "CUDA: Available ($gpu_name)"
            else
                print_info "CUDA: Not available (CPU mode)"
            fi
        else
            print_warning "PyTorch: Not installed"
        fi

        if python3 -c "import transformers" 2>/dev/null; then
            transformers_version=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
            print_success "Transformers: $transformers_version"
        else
            print_warning "Transformers: Not installed"
        fi
    fi

    # Check data files
    if [[ -f "data/train.csv" ]] && [[ -f "data/test.csv" ]]; then
        train_count=$(tail -n +2 data/train.csv | wc -l)
        test_count=$(tail -n +2 data/test.csv | wc -l)
        print_success "Data files: train.csv ($train_count samples), test.csv ($test_count samples)"
    else
        print_warning "Data files: Missing (need data/train.csv and data/test.csv)"
    fi

    # Check recent results
    if [[ -d "output" ]]; then
        latest_result=$(find output -name "*.csv" -o -name "*.pt" | head -1)
        if [[ -n "$latest_result" ]]; then
            print_success "Previous results: Available in output/"
        else
            print_info "Previous results: None found"
        fi
    else
        print_info "Output directory: Will be created"
    fi

    echo ""
}

# Main execution
main() {
    print_banner

    case "${1:-}" in
        --setup-only)
            check_location
            check_prerequisites
            check_data_files
            setup_directories
            setup_environment
            test_deberta_model
            print_success "🎉 Setup completed successfully!"
            print_info "Run '$0 --quick' to test the pipeline"
            print_info "Run '$0 --full' for full training"
            ;;

        --quick|--test)
            check_location
            check_prerequisites
            check_data_files
            setup_directories
            setup_environment
            run_detailed_data_verification
            test_deberta_model
            run_pipeline "quick"
            ;;

        --predict)
            if [[ -z "$2" ]]; then
                print_error "Model path required for prediction mode"
                show_help
                exit 1
            fi
            check_location
            check_prerequisites
            check_data_files
            setup_directories
            setup_environment
            run_detailed_data_verification
            run_pipeline "predict" "$2"
            ;;

        --status)
            check_location
            show_status
            ;;

        --help|-h)
            show_help
            ;;

        --full|"")
            check_location
            check_prerequisites
            check_data_files
            setup_directories
            setup_environment
            run_detailed_data_verification
            test_deberta_model
            run_pipeline "full"
            ;;

        *)
            print_error "Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}⏹️  Script interrupted by user${NC}"; exit 130' INT

# Run main function
main "$@"
