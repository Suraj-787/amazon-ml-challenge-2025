#!/usr/bin/env python3
"""
Main script for image-based price prediction with LoRA fine-tuning
Optimized for H100 80GB GPU with 4-hour training constraint

Usage:
    python image_model/main.py                    # Full training
    python image_model/main.py --quick            # Quick test run
    python image_model/main.py --predict-only     # Only make predictions
"""

import os
import sys
import argparse
import logging
import pandas as pd
import torch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_model.config import get_config
from image_model.trainer import ImagePriceTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_model/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Image-based Price Prediction with LoRA Fine-tuning"
    )
    
    # Mode
    parser.add_argument(
        '--predict-only',
        action='store_true',
        help='Only make predictions (skip training)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test run with reduced settings'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        default='openai/clip-vit-large-patch14',
        help='Base model name'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=16,
        help='LoRA rank (8, 16, or 32)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Batch size (default: 512 for H100)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate for LoRA'
    )
    parser.add_argument(
        '--lr-head',
        type=float,
        default=1e-3,
        help='Learning rate for regression head'
    )
    
    # Data configuration
    parser.add_argument(
        '--train-limit',
        type=int,
        default=35000,
        help='Limit training samples (default: 35000)'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        default='image_model/output',
        help='Output directory'
    )
    parser.add_argument(
        '--checkpoint',
        help='Checkpoint directory for prediction'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config_kwargs = {
        'base_model': args.model,
        'lora_r': args.lora_r,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'learning_rate_head': args.lr_head,
        'train_limit': args.train_limit,
        'output_dir': args.output_dir,
    }
    
    # Quick mode adjustments
    if args.quick:
        logger.info("Quick mode enabled - reducing settings for fast testing")
        config_kwargs.update({
            'batch_size': 64,
            'num_epochs': 2,
            'train_limit': 1000,
        })
    
    config = get_config(**config_kwargs)
    
    # Print configuration
    logger.info("="*60)
    logger.info("Image-based Price Prediction with LoRA")
    logger.info("="*60)
    logger.info(f"Base Model: {config.base_model}")
    logger.info(f"LoRA Rank: {config.lora_r}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Learning Rate (LoRA): {config.learning_rate}")
    logger.info(f"Learning Rate (Head): {config.learning_rate_head}")
    logger.info(f"Train Limit: {config.train_limit}")
    logger.info(f"Output Dir: {config.output_dir}")
    logger.info(f"Use BF16: {config.use_bf16}")
    logger.info(f"Use torch.compile: {config.use_torch_compile}")
    logger.info("="*60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This model requires GPU.")
        sys.exit(1)
    
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize trainer
    trainer = ImagePriceTrainer(config)
    
    if not args.predict_only:
        # Train model
        logger.info("\nStarting training...")
        checkpoint_dir = trainer.train()
        logger.info(f"\n✓ Training complete! Best model saved to: {checkpoint_dir}")
    else:
        # Use provided checkpoint
        if not args.checkpoint:
            logger.error("--checkpoint required for --predict-only mode")
            sys.exit(1)
        checkpoint_dir = args.checkpoint
        logger.info(f"Using checkpoint: {checkpoint_dir}")
    
    # Make predictions on test set
    logger.info("\nMaking predictions on test set...")
    predictions, sample_ids = trainer.predict(
        test_csv=config.test_csv,
        checkpoint_dir=checkpoint_dir
    )
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Sort by sample_id to match expected format
    submission = submission.sort_values('sample_id').reset_index(drop=True)
    
    # Save submission
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = os.path.join(config.output_dir, f'submission_{timestamp}.csv')
    submission.to_csv(submission_path, index=False)
    
    logger.info(f"\n✓ Submission saved to: {submission_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    logger.info(f"Mean price: ${submission['price'].mean():.2f}")
    logger.info(f"Median price: ${submission['price'].median():.2f}")
    
    logger.info("\n" + "="*60)
    logger.info("🎉 All done!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
