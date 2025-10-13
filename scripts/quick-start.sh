#!/bin/bash

# Quick Start Script for ML Challenge 2025
# Team: Monk Builders
# Members: Suraj Kumar, Himanshu Rai, Pratik Gautam, Impana KS

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
RED='\033[0;31m'
NC='\033[0m'

# Print banner
print_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                 ML Challenge 2025                        ║"
    echo "║              Multimodal Fusion Pipeline                  ║"
    echo "║                  Team: Monk Builders                     ║"
    echo "║      Suraj Kumar, Himanshu Rai, Pratik Gautam, Impana KS║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
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

# Check prerequisites
check_prerequisites() {
    print_step "1" "Checking prerequisites..."

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9 or newer."
        exit 1
    fi

    python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    print_success "Python $python_version detected"

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        print_step "1.1" "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        print_success "uv installed successfully"
    else
        print_success "uv is already available"
    fi

    # Check GPU (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected - will use GPU acceleration if available"
    else
        echo -e "${YELLOW}[INFO]${NC} No GPU detected - will use CPU mode (slower but functional)"
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

    print_success "Data files found!"
}

# Setup directories
setup_directories() {
    print_step "3" "Setting up directories..."
    mkdir -p output notebooks models
    print_success "Directories created!"
}

# Setup virtual environment
setup_environment() {
    print_step "4" "Setting up Python environment..."

    if [[ ! -d ".venv" ]]; then
        uv venv .venv --python python3
        print_success "Virtual environment created!"
    else
        print_success "Virtual environment already exists!"
    fi
}

# Show next steps
show_next_steps() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                     SETUP COMPLETE!                     ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    echo -e "${YELLOW}🚀 Next Steps:${NC}"
    echo "  ./scripts/setup-and-run.sh       # Full setup and run pipeline"
    echo "  ./scripts/setup-and-run.sh --help # Get help"
    echo ""
    echo -e "${BLUE}Manual commands:${NC}"
    echo "  source .venv/bin/activate        # Activate environment"
    echo "  python src/main.py               # Run pipeline"
    echo ""
    echo -e "${GREEN}✨ Ready to go! ✨${NC}"
}

# Main execution
main() {
    print_banner

    case "${1:-}" in
        --auto)
            check_prerequisites
            check_data_files
            setup_directories
            setup_environment
            show_next_steps
            ;;
        *)
            echo -e "${YELLOW}Run complete setup? (y/N):${NC}"
            read -r response
            if [[ $response =~ ^[Yy]$ ]]; then
                check_prerequisites
                check_data_files
                setup_directories
                setup_environment
                show_next_steps
            else
                echo "Setup cancelled."
            fi
            ;;
    esac
}

main "$@"
