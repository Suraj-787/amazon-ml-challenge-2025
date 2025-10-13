"""
Configuration file for DeBERTa price prediction model
Contains all hyperparameters and settings as specified in plan.md
"""

import torch


class ModelConfig:
    """Model configuration following the plan.md strategy"""

    # Model architecture
    MODEL_NAME = "microsoft/deberta-v3-large"  # 304M parameters - perfect for A10G
    MAX_LENGTH = 256  # Sequence length for tokenization
    DROPOUT = 0.1  # Dropout rate for regularization

    # Training parameters
    BATCH_SIZE = 16  # Base batch size
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16 * 2 = 32
    NUM_EPOCHS = 3  # As recommended in plan

    # Learning rates (different for encoder vs head as per plan)
    LEARNING_RATE_ENCODER = 1e-5  # Lower LR for pre-trained encoder
    LEARNING_RATE_HEAD = 1e-3  # Higher LR for regression head

    # Scheduler settings
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01  # For regularization

    # Loss function
    LOSS_TYPE = "mse_log"  # MSE in log space as recommended
    USE_SMAPE_LOSS = False  # Can be enabled for direct SMAPE optimization

    # Data preprocessing
    LOG_TRANSFORM_PRICES = True  # Critical for price prediction
    TEST_SIZE = 0.2  # Validation split

    # Training optimization
    GRADIENT_CLIP_NORM = 1.0  # Gradient clipping
    USE_MIXED_PRECISION = True  # FP16 for A10G

    # Device and performance
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # DataLoader workers
    PIN_MEMORY = True  # For GPU efficiency

    # Model saving
    SAVE_EVERY_EPOCH = True
    SAVE_BEST_MODEL = True
    EARLY_STOPPING_PATIENCE = None  # Disabled for now

    # Paths
    DATA_DIR = "data"
    OUTPUT_DIR = "output"

    @classmethod
    def get_regression_head_config(cls):
        """Get regression head architecture as per plan.md"""
        return {
            "input_size": 1024,  # DeBERTa-V3-Large hidden size
            "layer1_size": 512,
            "layer2_size": 256,
            "output_size": 1,
            "activation": "gelu",  # GELU activation as recommended
            "dropout": cls.DROPOUT,
        }

    @classmethod
    def get_optimizer_config(cls):
        """Get optimizer configuration with different LRs"""
        return {
            "encoder_lr": cls.LEARNING_RATE_ENCODER,
            "head_lr": cls.LEARNING_RATE_HEAD,
            "weight_decay": cls.WEIGHT_DECAY,
            "warmup_steps": cls.WARMUP_STEPS,
        }

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("🔧 DeBERTa-V3-Large Configuration")
        print("=" * 50)
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Max Length: {cls.MAX_LENGTH}")
        print(
            f"Batch Size: {cls.BATCH_SIZE} (effective: {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS})"
        )
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Encoder LR: {cls.LEARNING_RATE_ENCODER}")
        print(f"Head LR: {cls.LEARNING_RATE_HEAD}")
        print(f"Device: {cls.DEVICE}")
        print(f"Dropout: {cls.DROPOUT}")
        print(f"Weight Decay: {cls.WEIGHT_DECAY}")
        print("=" * 50)


class ExperimentConfig:
    """Configuration for different experimental settings"""

    # Quick test run (for debugging)
    QUICK_TEST = {
        "batch_size": 8,
        "num_epochs": 1,
        "max_samples": 1000,
        "max_length": 128,
    }

    # Production run (full training)
    PRODUCTION = {
        "batch_size": 16,
        "num_epochs": 3,
        "max_samples": None,
        "max_length": 256,
    }

    # High performance run (if you have more GPU memory)
    HIGH_PERFORMANCE = {
        "batch_size": 32,
        "num_epochs": 5,
        "max_samples": None,
        "max_length": 512,
        "model_name": "microsoft/deberta-v3-xlarge",  # 900M parameters
    }


class DataConfig:
    """Data processing configuration"""

    # Column names
    TEXT_COLUMN = "catalog_content"
    PRICE_COLUMN = "price"
    ID_COLUMN = "sample_id"
    IMAGE_COLUMN = "image_link"  # For future multimodal extension

    # Text preprocessing
    CLEAN_TEXT = True
    REMOVE_HTML = True
    NORMALIZE_UNICODE = True

    # Price preprocessing
    MIN_PRICE = 0.01  # Minimum valid price
    MAX_PRICE = 10000  # Maximum valid price (outlier threshold)

    # Data validation
    CHECK_DUPLICATES = True
    CHECK_MISSING_VALUES = True

    @classmethod
    def get_text_preprocessing_config(cls):
        """Get text preprocessing configuration"""
        return {
            "clean_text": cls.CLEAN_TEXT,
            "remove_html": cls.REMOVE_HTML,
            "normalize_unicode": cls.NORMALIZE_UNICODE,
        }


# Default configuration instance
config = ModelConfig()

# Export commonly used values
MODEL_NAME = ModelConfig.MODEL_NAME
BATCH_SIZE = ModelConfig.BATCH_SIZE
NUM_EPOCHS = ModelConfig.NUM_EPOCHS
LEARNING_RATE_ENCODER = ModelConfig.LEARNING_RATE_ENCODER
LEARNING_RATE_HEAD = ModelConfig.LEARNING_RATE_HEAD
MAX_LENGTH = ModelConfig.MAX_LENGTH
DEVICE = ModelConfig.DEVICE

if __name__ == "__main__":
    # Print configuration when run directly
    ModelConfig.print_config()
