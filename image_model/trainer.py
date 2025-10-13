"""
Training loop for image-based price prediction
Optimized for H100 80GB with 4-hour constraint
"""

import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple
import json

from .model import create_model
from .s3_dataset import S3ImageDataset, collate_fn

logger = logging.getLogger(__name__)


class ImagePriceTrainer:
    """Trainer for image-based price prediction with LoRA"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize model
        self.model = create_model(config).to(self.device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics history
        self.train_losses = []
        self.val_losses = []
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for LoRA and head"""
        # Separate parameters
        lora_params = []
        head_params = list(self.model.regression_head.parameters())
        
        if self.config.use_lora:
            # Get LoRA parameters
            for name, param in self.model.vision_model.named_parameters():
                if param.requires_grad:
                    lora_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {'params': head_params, 'lr': self.config.learning_rate_head},
        ]
        
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': self.config.learning_rate
            })
        
        optimizer = AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders"""
        logger.info("Creating datasets...")
        
        # Create full training dataset
        full_dataset = S3ImageDataset(
            csv_path=self.config.train_csv,
            s3_bucket=self.config.s3_bucket,
            s3_prefix=self.config.s3_train_prefix,
            image_size=self.config.image_size,
            use_log_transform=self.config.use_log_transform,
            aws_region=self.config.aws_region,
            limit=self.config.train_limit,
            augment=True,  # Augmentation for training
        )
        
        # Split into train and validation (90/10)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update validation dataset to not use augmentation
        val_dataset.dataset.augment = False
        val_dataset.dataset.transform = val_dataset.dataset._get_transforms(augment=False)
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (images, prices, sample_ids) in enumerate(pbar):
            images = images.to(self.device)
            prices = prices.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            loss = self.criterion(predictions, prices)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step}: "
                    f"loss={loss.item():.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc="Validation")
        
        for images, prices, sample_ids in pbar:
            images = images.to(self.device)
            prices = prices.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            loss = self.criterion(predictions, prices)
            
            total_loss += loss.item()
            
            # Store for metrics
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(prices.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate SMAPE
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets).flatten()
        
        # Convert back from log space if needed
        if self.config.use_log_transform:
            predictions = np.expm1(predictions)  # exp(x) - 1
            targets = np.expm1(targets)
        
        # SMAPE calculation
        smape = self._calculate_smape(targets, predictions)
        
        return avg_loss, smape
    
    def _calculate_smape(self, y_true, y_pred):
        """Calculate Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0.0
        return 100 * np.mean(diff)
    
    def train(self) -> str:
        """
        Main training loop
        
        Returns:
            Path to best model checkpoint
        """
        logger.info("="*60)
        logger.info("Starting Training")
        logger.info("="*60)
        
        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders()
        
        # Create optimizer
        optimizer = self._create_optimizer()
        
        # Create learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[self.config.learning_rate_head, self.config.learning_rate],
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy='cos',
        )
        
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_smape = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val SMAPE: {val_smape:.2f}%"
            )
            
            # Save checkpoint if best
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save model
                checkpoint_dir = os.path.join(self.config.output_dir, 'best_model')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                self.model.save_lora_weights(checkpoint_dir)
                self.model.save_regression_head(
                    os.path.join(checkpoint_dir, 'regression_head.pt')
                )
                
                logger.info(
                    f"✓ New best model saved! "
                    f"Val loss improved by {improvement:.4f}"
                )
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement for {self.patience_counter} epochs "
                    f"(patience: {self.config.early_stopping_patience})"
                )
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping triggered!")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        logger.info(f"\nTraining completed in {elapsed_time / 3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': float(self.best_val_loss),
            'total_epochs': epoch + 1,
            'training_time_hours': elapsed_time / 3600,
        }
        
        with open(os.path.join(self.config.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        return os.path.join(self.config.output_dir, 'best_model')
    
    @torch.no_grad()
    def predict(self, test_csv: str, checkpoint_dir: str) -> np.ndarray:
        """
        Make predictions on test set
        
        Args:
            test_csv: Path to test CSV
            checkpoint_dir: Directory with saved model weights
        
        Returns:
            predictions: Array of predicted prices
        """
        logger.info("Loading best model for prediction...")
        
        # Load weights
        self.model.load_lora_weights(checkpoint_dir)
        self.model.load_regression_head(
            os.path.join(checkpoint_dir, 'regression_head.pt')
        )
        self.model.eval()
        
        # Create test dataset
        test_dataset = S3ImageDataset(
            csv_path=test_csv,
            s3_bucket=self.config.s3_bucket,
            s3_prefix=self.config.s3_test_prefix,
            image_size=self.config.image_size,
            use_log_transform=False,  # Don't transform for test
            aws_region=self.config.aws_region,
            augment=False,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        logger.info(f"Predicting on {len(test_dataset)} test samples...")
        
        all_predictions = []
        all_sample_ids = []
        
        for images, _, sample_ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(self.device)
            
            predictions = self.model(images)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_sample_ids.extend(sample_ids)
        
        predictions = np.array(all_predictions).flatten()
        
        # Convert back from log space if needed
        if self.config.use_log_transform:
            predictions = np.expm1(predictions)
        
        # Ensure positive prices
        predictions = np.maximum(predictions, 0.01)
        
        logger.info(f"Predictions complete. Range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        
        return predictions, all_sample_ids
