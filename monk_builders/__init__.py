"""
ML Challenge 2025 - DeBERTa-V3-Large Price Prediction Package
Team Monk Builders

This package implements a state-of-the-art price prediction model using
DeBERTa-V3-Large following the strategy outlined in plan.md.

Main Components:
- deberta_price_predictor: Core model implementation
- data_utils: Data preprocessing and validation utilities
- evaluation: Model evaluation and metrics calculation
- config: Configuration management
"""

from .deberta_price_predictor import (
    PricePredictionTrainer,
    DeBERTaPricePredictor,
    SMAPELoss,
    PriceDataset,
    load_model,
)

from .data_utils import (
    DataPreprocessor,
    DataValidator,
    load_and_preprocess_data,
    split_training_data,
)

from .evaluation import (
    calculate_smape,
    calculate_mape,
    calculate_wmape,
    comprehensive_evaluation,
    evaluate_model_predictions,
    compare_models,
)

from .config import (
    ModelConfig,
    ExperimentConfig,
    DataConfig,
    MODEL_NAME,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE_ENCODER,
    LEARNING_RATE_HEAD,
    MAX_LENGTH,
    DEVICE,
)

__version__ = "1.0.0"
__author__ = "Team Monk Builders"
__description__ = "DeBERTa-V3-Large Price Prediction Model"

# Main exports
__all__ = [
    # Core model
    "PricePredictionTrainer",
    "DeBERTaPricePredictor",
    "SMAPELoss",
    "PriceDataset",
    "load_model",
    # Data utilities
    "DataPreprocessor",
    "DataValidator",
    "load_and_preprocess_data",
    "split_training_data",
    # Evaluation
    "calculate_smape",
    "calculate_mape",
    "calculate_wmape",
    "comprehensive_evaluation",
    "evaluate_model_predictions",
    "compare_models",
    # Configuration
    "ModelConfig",
    "ExperimentConfig",
    "DataConfig",
    "MODEL_NAME",
    "BATCH_SIZE",
    "NUM_EPOCHS",
    "LEARNING_RATE_ENCODER",
    "LEARNING_RATE_HEAD",
    "MAX_LENGTH",
    "DEVICE",
]
