import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriceDataset(Dataset):
    """Dataset for product price prediction using text descriptions"""

    def __init__(
        self,
        texts: List[str],
        prices: Optional[List[float]] = None,
        tokenizer=None,
        max_length: int = 256,
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
        self, model_name: str = "microsoft/deberta-v3-large", dropout: float = 0.1
    ):
        super(DeBERTaPricePredictor, self).__init__()

        # Load pre-trained DeBERTa model
        self.deberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.deberta.config.hidden_size  # 1024 for large

        # Regression head with multiple layers
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(self.hidden_size, 512)
        self.activation1 = nn.GELU()

        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(512, 256)
        self.activation2 = nn.GELU()

        self.dropout3 = nn.Dropout(dropout)
        self.output = nn.Linear(256, 1)

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


class SMAPELoss(nn.Module):
    """Symmetric Mean Absolute Percentage Error loss"""

    def __init__(self, epsilon: float = 1e-8):
        super(SMAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Convert back from log space for SMAPE calculation
        pred_price = torch.exp(pred) - 1
        target_price = torch.exp(target) - 1

        numerator = torch.abs(pred_price - target_price)
        denominator = (
            torch.abs(pred_price) + torch.abs(target_price)
        ) / 2 + self.epsilon

        smape = 100 * torch.mean(numerator / denominator)
        return smape


class PricePredictionTrainer:
    """Trainer class for DeBERTa price prediction model"""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        max_length: int = 256,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 2,
        learning_rate_encoder: float = 1e-5,
        learning_rate_head: float = 1e-3,
        num_epochs: int = 3,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize model
        self.model = DeBERTaPricePredictor(model_name, dropout).to(self.device)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.smape_loss = SMAPELoss()

        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

    def prepare_data(
        self, df: pd.DataFrame, test_size: float = 0.2
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
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, val_loader

    def setup_optimizer_and_scheduler(self, train_loader):
        """Setup optimizer with different learning rates for encoder and head"""

        # Separate parameters
        encoder_params = []
        head_params = []

        for name, param in self.model.named_parameters():
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

        return optimizer, scheduler

    def calculate_smape(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Calculate SMAPE metric"""
        # Convert back from log space
        pred_prices = torch.exp(predictions.cpu()) - 1
        target_prices = torch.exp(targets.cpu()) - 1

        pred_prices = pred_prices.numpy()
        target_prices = target_prices.numpy()

        # Calculate SMAPE
        numerator = np.abs(pred_prices - target_prices)
        denominator = (np.abs(pred_prices) + np.abs(target_prices)) / 2

        # Avoid division by zero
        valid_mask = denominator > 1e-8
        if valid_mask.sum() == 0:
            return 0.0

        smape = 100 * np.mean(numerator[valid_mask] / denominator[valid_mask])
        return smape

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

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
        train_smape = self.calculate_smape(
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
        val_smape = self.calculate_smape(
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
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()

        # Create dataset
        dataset = PriceDataset(texts, None, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
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


def load_model(model_path: str, device: str = None):
    """Load trained model from checkpoint"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Initialize model
    model = DeBERTaPricePredictor(
        model_name=config["model_name"], dropout=config["dropout"]
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = checkpoint["tokenizer"]

    return model, tokenizer


def main():
    """Example usage"""

    # Load data
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Initialize trainer
    trainer = PricePredictionTrainer(
        model_name="microsoft/deberta-v3-large",
        batch_size=16,
        gradient_accumulation_steps=2,
        num_epochs=3,
        learning_rate_encoder=1e-5,
        learning_rate_head=1e-3,
    )

    # Train model
    best_model_path = trainer.train(train_df)

    # Make predictions on test set
    test_texts = test_df["catalog_content"].tolist()
    predictions = trainer.predict(test_texts, best_model_path)

    # Create submission
    submission = pd.DataFrame({"sample_id": test_df["sample_id"], "price": predictions})

    submission.to_csv("output/submission.csv", index=False)
    print("Predictions saved to output/submission.csv")


if __name__ == "__main__":
    main()
