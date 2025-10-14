#!/usr/bin/env python3
"""
Comprehensive DeBERTa-V3-Large Price Prediction Model
Team Monk Builders - ML Challenge 2025

This script implements the complete pipeline following docs/plan.md strategy:
- DeBERTa-V3-Large (304M parameters) for text understanding
- Log-space training for price prediction
- Differential learning rates for encoder vs head
- SMAPE evaluation metric
- Complete data preprocessing and validation

Usage:
    python main.py                    # Full training + prediction
    python main.py --quick-test      # Quick test run
    python main.py --predict-only --model-path output/best_model.pt
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import re
import unicodedata
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class ModelConfig:
    """Model configuration following the docs/plan.md strategy"""

    # Model architecture (using base model for speed and memory efficiency)
    MODEL_NAME = (
        "microsoft/deberta-v3-base"  # 86M parameters - much faster and memory efficient
    )
    MAX_LENGTH = 256  # Sequence length for tokenization
    DROPOUT = 0.1  # Dropout rate for regularization

    # Training parameters (optimized for multi-GPU)
    BATCH_SIZE = 16  # Larger batch size with base model
    GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 16 * 2 = 32
    NUM_EPOCHS = 3  # As recommended in plan

    # Learning rates (different for encoder vs head as per plan)
    LEARNING_RATE_ENCODER = 1e-5  # Lower LR for pre-trained encoder
    LEARNING_RATE_HEAD = 1e-3  # Higher LR for regression head

    # Scheduler settings
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01  # For regularization

    # Training optimization
    GRADIENT_CLIP_NORM = 1.0  # Gradient clipping

    # Data preprocessing
    TEST_SIZE = 0.2  # Validation split
    MIN_PRICE = 0.01  # Minimum valid price
    MAX_PRICE = 10000.0  # Maximum valid price

    # Device and performance (multi-GPU optimized)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Enable multiprocessing for speed
    PIN_MEMORY = True  # Enable pin memory for GPU efficiency
    USE_MULTI_GPU = torch.cuda.device_count() > 1  # Auto-detect multiple GPUs


# =============================================================================
# DATA PREPROCESSING
# =============================================================================


class DataPreprocessor:
    """Data preprocessing utilities for product catalog data"""

    def __init__(
        self,
        min_price: float = ModelConfig.MIN_PRICE,
        max_price: float = ModelConfig.MAX_PRICE,
        clean_text: bool = True,
        remove_html: bool = True,
        normalize_unicode: bool = True,
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.clean_text = clean_text
        self.remove_html = remove_html
        self.normalize_unicode = normalize_unicode

    def clean_catalog_content(self, text: str) -> str:
        """Clean product catalog content"""
        if pd.isna(text) or text == "":
            return ""

        text = str(text)

        # Normalize unicode characters
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKD", text)

        # Remove HTML tags if present
        if self.remove_html:
            text = re.sub(r"<[^>]+>", " ", text)

        if self.clean_text:
            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove special characters but keep important ones
            text = re.sub(r"[^\w\s\.\,\!\?\:\;\-\(\)\$\%\&]", " ", text)

            # Clean up common patterns
            text = re.sub(r"\s*\n\s*", " ", text)  # Replace newlines with space
            text = re.sub(r"\s*\t\s*", " ", text)  # Replace tabs with space
            text = re.sub(r"\s{2,}", " ", text)  # Multiple spaces to single

            # Strip leading/trailing whitespace
            text = text.strip()

        return text

    def validate_price(self, price: float) -> bool:
        """Validate if price is within acceptable range"""
        if pd.isna(price):
            return False
        if price < self.min_price or price > self.max_price:
            return False
        return True

    def preprocess_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess training data"""
        logger.info("Preprocessing training data...")
        df = df.copy()

        original_count = len(df)
        logger.info(f"Original training samples: {original_count}")

        # Check required columns
        required_cols = ["catalog_content", "price", "sample_id"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove rows with missing data
        df = df.dropna(subset=["catalog_content", "price"])

        # Clean catalog content
        logger.info("Cleaning catalog content...")
        df["catalog_content"] = df["catalog_content"].apply(self.clean_catalog_content)

        # Remove empty content after cleaning
        df = df[df["catalog_content"].str.len() > 0]

        # Filter valid prices
        valid_price_mask = df["price"].apply(self.validate_price)
        df = df[valid_price_mask]

        # Remove duplicates based on catalog_content
        df = df.drop_duplicates(subset=["catalog_content"], keep="first")

        final_count = len(df)
        logger.info(f"Final training samples: {final_count}")
        logger.info(
            f"Removed {original_count - final_count} samples during preprocessing"
        )

        return df.reset_index(drop=True)

    def preprocess_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess test data"""
        logger.info("Preprocessing test data...")
        df = df.copy()

        # Handle missing catalog_content (fill with empty string)
        df["catalog_content"] = df["catalog_content"].fillna("")

        # Clean catalog content
        df["catalog_content"] = df["catalog_content"].apply(self.clean_catalog_content)

        return df.reset_index(drop=True)


# =============================================================================
# DATASET AND MODEL
# =============================================================================


class PriceDataset(Dataset):
    """Dataset for product price prediction using text descriptions"""

    def __init__(
        self,
        texts: List[str],
        prices: Optional[List[float]] = None,
        tokenizer=None,
        max_length: int = ModelConfig.MAX_LENGTH,
    ):
        self.texts = texts
        self.prices = prices
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        if self.prices is not None:
            # Log transform price (critical for price prediction)
            price = float(self.prices[idx])
            log_price = np.log(price + 1)  # +1 to handle zero prices
            item["price"] = torch.tensor(log_price, dtype=torch.float32)

        return item


class DeBERTaPricePredictor(nn.Module):
    """DeBERTa-V3-Large based price prediction model"""

    def __init__(
        self,
        model_name: str = ModelConfig.MODEL_NAME,
        dropout: float = ModelConfig.DROPOUT,
    ):
        super(DeBERTaPricePredictor, self).__init__()

        # Load pre-trained DeBERTa model (base model for speed)
        self.deberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.deberta.config.hidden_size  # 768 for base

        # Regression head optimized for base model
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(self.hidden_size, 384)  # Smaller for base model
        self.activation1 = nn.GELU()

        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(384, 192)
        self.activation2 = nn.GELU()

        self.dropout3 = nn.Dropout(dropout)
        self.output = nn.Linear(192, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize regression head weights"""
        for module in [self.dense1, self.dense2, self.output]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        # Get DeBERTa embeddings
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[
            :, 0, :
        ]  # Shape: [batch_size, hidden_size]

        # Pass through regression head
        x = self.dropout1(cls_output)
        x = self.dense1(x)
        x = self.activation1(x)

        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.activation2(x)

        x = self.dropout3(x)
        output = self.output(x)

        return output.squeeze(-1)  # Shape: [batch_size]


# =============================================================================
# EVALUATION METRICS
# =============================================================================


def calculate_smape(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE) - Challenge metric"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure no negative values
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon

    smape = 100 * np.mean(numerator / denominator)
    return smape


def comprehensive_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
    """Perform comprehensive evaluation with all relevant metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    results = {}

    # Basic info
    results["sample_count"] = len(y_true)
    results["price_range_true"] = (float(np.min(y_true)), float(np.max(y_true)))
    results["price_range_pred"] = (float(np.min(y_pred)), float(np.max(y_pred)))

    # Main metrics
    results["smape"] = calculate_smape(y_true, y_pred)
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    results["r2"] = r2_score(y_true, y_pred)

    # Log-space metrics
    log_true = np.log(y_true + 1)
    log_pred = np.log(y_pred + 1)
    results["log_mae"] = mean_absolute_error(log_true, log_pred)
    results["log_rmse"] = np.sqrt(mean_squared_error(log_true, log_pred))
    results["log_r2"] = r2_score(log_true, log_pred)

    return results


def print_evaluation_report(results: Dict[str, any]) -> None:
    """Print a formatted evaluation report"""
    print("\n📊 Model Evaluation Report")
    print("=" * 60)

    # Basic info
    print(f"Sample Count: {results['sample_count']:,}")
    print(
        f"True Price Range: ${results['price_range_true'][0]:.2f} - ${results['price_range_true'][1]:.2f}"
    )
    print(
        f"Pred Price Range: ${results['price_range_pred'][0]:.2f} - ${results['price_range_pred'][1]:.2f}"
    )

    # Main metrics
    print(f"\n🎯 Primary Metrics:")
    print(f"  SMAPE (Challenge Metric): {results['smape']:.2f}%")
    print(f"  MAE: ${results['mae']:.2f}")
    print(f"  RMSE: ${results['rmse']:.2f}")
    print(f"  R²: {results['r2']:.4f}")

    # Log-space metrics
    print(f"\n📊 Log-Space Metrics:")
    print(f"  Log MAE: {results['log_mae']:.4f}")
    print(f"  Log RMSE: {results['log_rmse']:.4f}")
    print(f"  Log R²: {results['log_r2']:.4f}")

    print("=" * 60)


# =============================================================================
# TRAINER CLASS
# =============================================================================


class PricePredictionTrainer:
    """Trainer class for DeBERTa price prediction model"""

    def __init__(
        self,
        model_name: str = ModelConfig.MODEL_NAME,
        max_length: int = ModelConfig.MAX_LENGTH,
        batch_size: int = ModelConfig.BATCH_SIZE,
        gradient_accumulation_steps: int = ModelConfig.GRADIENT_ACCUMULATION_STEPS,
        learning_rate_encoder: float = ModelConfig.LEARNING_RATE_ENCODER,
        learning_rate_head: float = ModelConfig.LEARNING_RATE_HEAD,
        num_epochs: int = ModelConfig.NUM_EPOCHS,
        warmup_steps: int = ModelConfig.WARMUP_STEPS,
        weight_decay: float = ModelConfig.WEIGHT_DECAY,
        dropout: float = ModelConfig.DROPOUT,
        device: str = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate_encoder = learning_rate_encoder
        self.learning_rate_head = learning_rate_head
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device or ModelConfig.DEVICE

        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model
        logger.info(f"Loading model: {model_name}")
        self.model = DeBERTaPricePredictor(model_name, dropout)

        # Multi-GPU setup
        if ModelConfig.USE_MULTI_GPU and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)

        # Loss function
        self.mse_loss = nn.MSELoss()

        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

    def prepare_data(
        self, df: pd.DataFrame, test_size: float = ModelConfig.TEST_SIZE
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare train and validation dataloaders"""

        texts = df["catalog_content"].tolist()
        prices = df["price"].tolist()

        # Split data
        train_texts, val_texts, train_prices, val_prices = train_test_split(
            texts, prices, test_size=test_size, random_state=42
        )

        # Create datasets
        train_dataset = PriceDataset(
            train_texts, train_prices, self.tokenizer, self.max_length
        )
        val_dataset = PriceDataset(
            val_texts, val_prices, self.tokenizer, self.max_length
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=ModelConfig.NUM_WORKERS,
            pin_memory=ModelConfig.PIN_MEMORY,
            drop_last=True,  # Drop incomplete batches to ensure consistent memory usage
            persistent_workers=True
            if ModelConfig.NUM_WORKERS > 0
            else False,  # Keep workers alive for speed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=ModelConfig.NUM_WORKERS,
            pin_memory=ModelConfig.PIN_MEMORY,
            drop_last=False,  # Keep all validation samples
            persistent_workers=True
            if ModelConfig.NUM_WORKERS > 0
            else False,  # Keep workers alive for speed
        )

        logger.info(
            f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )
        return train_loader, val_loader

    def setup_optimizer_and_scheduler(self, train_loader):
        """Setup optimizer with different learning rates for encoder and head"""

        # Separate parameters (handle DataParallel wrapper)
        encoder_params = []
        head_params = []

        model_to_check = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for name, param in model_to_check.named_parameters():
            if "deberta" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        # Setup optimizer with different learning rates
        optimizer = AdamW(
            [
                {"params": encoder_params, "lr": self.learning_rate_encoder},
                {"params": head_params, "lr": self.learning_rate_head},
            ],
            weight_decay=self.weight_decay,
        )

        # Setup scheduler
        total_steps = (
            len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(f"Total training steps: {total_steps}")
        return optimizer, scheduler

    def calculate_smape_tensor(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Calculate SMAPE metric from tensors"""
        # Convert back from log space
        pred_prices = torch.exp(predictions.cpu()) - 1
        target_prices = torch.exp(targets.cpu()) - 1

        pred_prices = pred_prices.numpy()
        target_prices = target_prices.numpy()

        return calculate_smape(target_prices, pred_prices)

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            prices = batch["price"].to(self.device)

            # Forward pass
            predictions = self.model(input_ids, attention_mask)

            # Calculate loss (MSE in log space)
            loss = self.mse_loss(predictions, prices)

            # Normalize loss by gradient accumulation steps
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=ModelConfig.GRADIENT_CLIP_NORM
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # More memory efficient

                # Clear cache less frequently with base model
                if (step + 1) % 500 == 0:
                    torch.cuda.empty_cache()

            total_loss += loss.item() * self.gradient_accumulation_steps
            all_predictions.extend(predictions.detach().cpu().tolist())
            all_targets.extend(prices.detach().cpu().tolist())

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr_enc": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "lr_head": f"{optimizer.param_groups[1]['lr']:.2e}",
                }
            )

        avg_loss = total_loss / len(train_loader)
        train_smape = self.calculate_smape_tensor(
            torch.tensor(all_predictions), torch.tensor(all_targets)
        )

        return avg_loss, train_smape

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                prices = batch["price"].to(self.device)

                # Forward pass
                predictions = self.model(input_ids, attention_mask)

                # Calculate loss
                loss = self.mse_loss(predictions, prices)

                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().tolist())
                all_targets.extend(prices.cpu().tolist())

        avg_loss = total_loss / len(val_loader)
        val_smape = self.calculate_smape_tensor(
            torch.tensor(all_predictions), torch.tensor(all_targets)
        )

        return avg_loss, val_smape

    def train(self, train_df: pd.DataFrame, output_dir: str = "output"):
        """Main training loop"""

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data
        logger.info("Preparing data...")
        train_loader, val_loader = self.prepare_data(train_df)

        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler(train_loader)

        best_val_smape = float("inf")
        best_model_path = os.path.join(output_dir, "best_model.pt")

        logger.info("Starting training...")
        logger.info(f"Model: {self.model_name}")
        logger.info(
            f"Effective batch size: {self.batch_size * self.gradient_accumulation_steps}"
        )
        logger.info(
            f"Encoder LR: {self.learning_rate_encoder}, Head LR: {self.learning_rate_head}"
        )

        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_smape = self.train_epoch(
                train_loader, optimizer, scheduler, epoch
            )

            # Validation
            val_loss, val_smape = self.validate_epoch(val_loader)

            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train SMAPE: {train_smape:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val SMAPE: {val_smape:.2f}%"
            )

            # Save best model
            if val_smape < best_val_smape:
                best_val_smape = val_smape
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "tokenizer": self.tokenizer,
                        "config": {
                            "model_name": self.model_name,
                            "max_length": self.max_length,
                            "dropout": self.dropout,
                        },
                        "val_smape": val_smape,
                    },
                    best_model_path,
                )
                logger.info(f"Saved best model with SMAPE: {val_smape:.2f}%")

            # Save checkpoint every epoch
            checkpoint_path = os.path.join(
                output_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_smape": val_smape,
                },
                checkpoint_path,
            )

        logger.info(f"Training completed! Best validation SMAPE: {best_val_smape:.2f}%")
        return best_model_path

    def predict(self, texts: List[str], model_path: str = None) -> np.ndarray:
        """Make predictions on new data"""

        if model_path:
            # Load saved model
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

        # Create dataset
        dataset = PriceDataset(texts, None, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=ModelConfig.NUM_WORKERS,
            pin_memory=ModelConfig.PIN_MEMORY,
            persistent_workers=True if ModelConfig.NUM_WORKERS > 0 else False,
        )

        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                pred = self.model(input_ids, attention_mask)

                # Convert back from log space
                pred_prices = torch.exp(pred.cpu()) - 1
                predictions.extend(pred_prices.numpy())

        return np.array(predictions)


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def check_gpu():
    """Check GPU availability and print info"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU Available: {gpu_name} ({gpu_memory:.1f}GB)")
        return "cuda"
    else:
        logger.info("GPU not available, using CPU")
        return "cpu"


def load_data(data_dir: str = "data"):
    """Load training and test data"""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")

    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)

    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    # Basic validation
    assert "catalog_content" in train_df.columns, (
        "catalog_content column missing in training data"
    )
    assert "price" in train_df.columns, "price column missing in training data"
    assert "sample_id" in train_df.columns, "sample_id column missing in training data"

    assert "catalog_content" in test_df.columns, (
        "catalog_content column missing in test data"
    )
    assert "sample_id" in test_df.columns, "sample_id column missing in test data"

    # Basic price statistics
    price_stats = train_df["price"].describe()
    logger.info(f"Price statistics:\n{price_stats}")

    return train_df, test_df


def create_output_dir(base_dir: str = "output"):
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"deberta_training_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def main():
    """Main training and prediction pipeline"""

    parser = argparse.ArgumentParser(description="DeBERTa Price Prediction Training")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--output-dir", default="output", help="Output directory path")
    parser.add_argument(
        "--model-name", default=ModelConfig.MODEL_NAME, help="Model name"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=ModelConfig.MAX_LENGTH,
        help="Max sequence length",
    )
    parser.add_argument(
        "--batch-size", type=int, default=ModelConfig.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=ModelConfig.GRADIENT_ACCUMULATION_STEPS,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate-encoder",
        type=float,
        default=ModelConfig.LEARNING_RATE_ENCODER,
        help="Encoder learning rate",
    )
    parser.add_argument(
        "--learning-rate-head",
        type=float,
        default=ModelConfig.LEARNING_RATE_HEAD,
        help="Head learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=ModelConfig.NUM_EPOCHS,
        help="Number of epochs",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=ModelConfig.WARMUP_STEPS,
        help="Warmup steps",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=ModelConfig.WEIGHT_DECAY,
        help="Weight decay",
    )
    parser.add_argument(
        "--dropout", type=float, default=ModelConfig.DROPOUT, help="Dropout rate"
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only make predictions with existing model",
    )
    parser.add_argument(
        "--model-path", help="Path to trained model for prediction only"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test run with reduced parameters",
    )

    args = parser.parse_args()

    # Setup
    logger.info("🚀 Starting DeBERTa-V3-Large Price Prediction Pipeline")
    logger.info("=" * 60)

    # Apply quick test modifications if requested
    if args.quick_test:
        logger.info("Quick test mode enabled - reducing parameters for fast testing")
        args.batch_size = 8
        args.num_epochs = 1
        args.max_length = 128
        args.warmup_steps = 100
        args.gradient_accumulation_steps = 1

    # Check GPU
    device = check_gpu()

    # Load raw data
    logger.info("Loading raw data...")
    train_df_raw, test_df_raw = load_data(args.data_dir)

    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    train_df = preprocessor.preprocess_training_data(train_df_raw)
    test_df = preprocessor.preprocess_test_data(test_df_raw)

    # Create configuration dictionary
    config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate_encoder": args.learning_rate_encoder,
        "learning_rate_head": args.learning_rate_head,
        "num_epochs": args.num_epochs,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "device": device,
    }

    # Create output directory
    if not args.predict_only:
        output_dir = create_output_dir(args.output_dir)
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    try:
        if args.predict_only:
            # Prediction only mode
            if not args.model_path:
                raise ValueError("Model path required for prediction-only mode")

            logger.info("Running in prediction-only mode")

            # Initialize trainer for prediction
            trainer = PricePredictionTrainer(
                model_name=config["model_name"],
                max_length=config["max_length"],
                batch_size=config["batch_size"],
                device=config["device"],
            )

            # Make predictions
            test_texts = test_df["catalog_content"].tolist()
            logger.info(f"Making predictions on {len(test_texts)} test samples...")
            predictions = trainer.predict(test_texts, args.model_path)

            # Create submission
            submission = pd.DataFrame(
                {"sample_id": test_df["sample_id"], "price": predictions}
            )
            submission_path = os.path.join(output_dir, "submission.csv")
            submission.to_csv(submission_path, index=False)

            logger.info(f"Predictions saved to: {submission_path}")

        else:
            # Full training + prediction pipeline
            logger.info("Running full training + prediction pipeline")

            # Print configuration
            logger.info(f"Configuration:")
            logger.info(f"  Model: {config['model_name']}")
            logger.info(
                f"  Batch size: {config['batch_size']} (effective: {config['batch_size'] * config['gradient_accumulation_steps']})"
            )
            logger.info(f"  Epochs: {config['num_epochs']}")
            logger.info(f"  Max length: {config['max_length']}")
            logger.info(f"  Encoder LR: {config['learning_rate_encoder']}")
            logger.info(f"  Head LR: {config['learning_rate_head']}")

            # Initialize trainer
            trainer = PricePredictionTrainer(
                model_name=config["model_name"],
                max_length=config["max_length"],
                batch_size=config["batch_size"],
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                learning_rate_encoder=config["learning_rate_encoder"],
                learning_rate_head=config["learning_rate_head"],
                num_epochs=config["num_epochs"],
                warmup_steps=config["warmup_steps"],
                weight_decay=config["weight_decay"],
                dropout=config["dropout"],
                device=config["device"],
            )

            # Train model
            logger.info("Starting training...")
            best_model_path = trainer.train(train_df, output_dir)

            # Generate predictions on test data
            logger.info("Generating predictions on test data...")
            test_texts = test_df["catalog_content"].tolist()
            predictions = trainer.predict(test_texts, best_model_path)

            # Create submission
            submission = pd.DataFrame(
                {"sample_id": test_df["sample_id"], "price": predictions}
            )
            submission_path = os.path.join(output_dir, "submission.csv")
            submission.to_csv(submission_path, index=False)

            # Log prediction statistics
            pred_stats = pd.Series(predictions).describe()
            logger.info(f"Prediction statistics:\n{pred_stats}")

            # Evaluate on validation set (using a small portion for quick evaluation)
            if len(train_df) > 1000:
                eval_df = train_df.sample(n=min(1000, len(train_df)), random_state=42)
                eval_texts = eval_df["catalog_content"].tolist()
                eval_predictions = trainer.predict(eval_texts, best_model_path)
                eval_true = eval_df["price"].values

                # Calculate evaluation metrics
                results = comprehensive_evaluation(eval_true, eval_predictions)
                print_evaluation_report(results)

                # Save evaluation results
                with open(
                    os.path.join(output_dir, "evaluation_results.json"), "w"
                ) as f:
                    # Convert numpy types for JSON serialization
                    results_json = {}
                    for key, value in results.items():
                        if isinstance(value, np.ndarray):
                            results_json[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating)):
                            results_json[key] = float(value)
                        else:
                            results_json[key] = value
                    json.dump(results_json, f, indent=2)

        logger.info("=" * 60)
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"📁 Output directory: {output_dir}")
        if not args.predict_only:
            logger.info(f"🏆 Best model: {best_model_path}")
        logger.info(f"📊 Predictions: {submission_path}")

    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {str(e)}")
        raise

    logger.info("🎉 DeBERTa Price Prediction Complete!")


if __name__ == "__main__":
    main()
