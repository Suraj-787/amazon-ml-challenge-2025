#!/usr/bin/env python3
"""
Improved DeBERTa Price Prediction Model with Feature Engineering
Implements all improvements from docs/plan.md:
1. Feature Engineering - Extract structured features from catalog_content
2. Multimodal Model - Combine text embeddings with numerical features
3. Mean Pooling - Option to use mean pooling instead of [CLS] token
4. K-Fold Cross-Validation - More robust training and evaluation

Usage:
    python src/main_improved.py                    # Full training with k-fold CV
    python src/main_improved.py --no-kfold        # Single train/val split
    python src/main_improved.py --use-mean-pooling # Use mean pooling
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
from sklearn.model_selection import KFold
import argparse
from datetime import datetime
import json
import gc

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import DataPreprocessor
from src.deberta_price_predictor import PricePredictionTrainer
from src.evaluation import evaluate_model_predictions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_numerical_feature_columns(df: pd.DataFrame) -> list:
    """Get list of numerical feature columns from preprocessed dataframe"""
    feature_cols = []
    
    # Add scaled numerical features
    if 'value' in df.columns:
        feature_cols.append('value')
    if 'pack_count' in df.columns:
        feature_cols.append('pack_count')
    
    # Add one-hot encoded unit features
    unit_cols = [col for col in df.columns if col.startswith('unit_')]
    feature_cols.extend(unit_cols)
    
    return feature_cols


def train_with_kfold(
    train_df: pd.DataFrame,
    model_name: str = "microsoft/deberta-v3-base",
    n_folds: int = 5,
    use_mean_pooling: bool = False,
    output_dir: str = "output",
):
    """
    Train model with k-fold cross-validation
    This provides more robust evaluation and better final predictions
    """
    logger.info(f"Starting {n_folds}-fold cross-validation training")
    
    # Get numerical feature columns
    numerical_feature_cols = get_numerical_feature_columns(train_df)
    num_numerical_features = len(numerical_feature_cols)
    
    logger.info(f"Using {num_numerical_features} numerical features: {numerical_feature_cols}")
    
    # Create k-fold splits
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    best_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Fold {fold + 1}/{n_folds}")
        logger.info(f"{'='*60}")
        
        # Split data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Create fold output directory
        fold_output_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Initialize trainer (optimized for H100 80GB)
        trainer = PricePredictionTrainer(
            model_name=model_name,
            batch_size=64,  # Increased for H100
            gradient_accumulation_steps=1,  # Not needed with H100
            num_epochs=3,  # 3 epochs per fold is optimal
            learning_rate_encoder=1e-5,
            learning_rate_head=1e-3,
            num_numerical_features=num_numerical_features,
            use_mean_pooling=use_mean_pooling,
        )
        
        # Train on this fold
        best_model_path = trainer.train(
            fold_train_df, 
            numerical_feature_cols, 
            fold_output_dir
        )
        
        # Evaluate on validation fold
        val_texts = fold_val_df["catalog_content"].tolist()
        val_prices = fold_val_df["price"].values
        val_num_feats = fold_val_df[numerical_feature_cols].values if numerical_feature_cols else None
        
        val_predictions = trainer.predict(val_texts, val_num_feats, best_model_path)
        
        # Calculate metrics
        from src.evaluation import comprehensive_evaluation
        fold_metrics = comprehensive_evaluation(val_prices, val_predictions)
        
        logger.info(f"Fold {fold + 1} Validation SMAPE: {fold_metrics['smape']:.2f}%")
        
        fold_results.append({
            'fold': fold + 1,
            'smape': fold_metrics['smape'],
            'mae': fold_metrics['mae'],
            'rmse': fold_metrics['rmse'],
            'r2': fold_metrics['r2'],
            'model_path': best_model_path
        })
        
        best_models.append(best_model_path)
        
        # Clean up GPU memory before next fold
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"GPU memory cleared after fold {fold + 1}")
    
    # Calculate average metrics across folds
    avg_smape = np.mean([r['smape'] for r in fold_results])
    std_smape = np.std([r['smape'] for r in fold_results])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"K-Fold Cross-Validation Results")
    logger.info(f"{'='*60}")
    logger.info(f"Average SMAPE: {avg_smape:.2f}% ± {std_smape:.2f}%")
    logger.info(f"Average MAE: ${np.mean([r['mae'] for r in fold_results]):.2f}")
    logger.info(f"Average RMSE: ${np.mean([r['rmse'] for r in fold_results]):.2f}")
    logger.info(f"Average R²: {np.mean([r['r2'] for r in fold_results]):.4f}")
    
    # Save fold results
    with open(os.path.join(output_dir, 'kfold_results.json'), 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'average_smape': float(avg_smape),
            'std_smape': float(std_smape),
        }, f, indent=2)
    
    return best_models, fold_results


def predict_with_ensemble(
    test_df: pd.DataFrame,
    model_paths: list,
    numerical_feature_cols: list,
    model_name: str = "microsoft/deberta-v3-base",
    use_mean_pooling: bool = False,
):
    """
    Make predictions using ensemble of k-fold models
    Average predictions from all folds for more robust results
    """
    logger.info(f"Making ensemble predictions with {len(model_paths)} models")
    
    test_texts = test_df["catalog_content"].tolist()
    test_num_feats = test_df[numerical_feature_cols].values if numerical_feature_cols else None
    num_numerical_features = len(numerical_feature_cols) if numerical_feature_cols else 0
    
    all_predictions = []
    
    for i, model_path in enumerate(model_paths):
        logger.info(f"Predicting with model {i + 1}/{len(model_paths)}")
        
        # Initialize trainer (optimized for H100)
        trainer = PricePredictionTrainer(
            model_name=model_name,
            batch_size=64,  # Increased for H100
            num_numerical_features=num_numerical_features,
            use_mean_pooling=use_mean_pooling,
        )
        
        # Make predictions
        predictions = trainer.predict(test_texts, test_num_feats, model_path)
        all_predictions.append(predictions)
    
    # Average predictions across all folds
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    logger.info("Ensemble predictions completed")
    return ensemble_predictions


def main():
    parser = argparse.ArgumentParser(description="Improved DeBERTa Price Prediction")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="output/improved", help="Output directory")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large", 
                       help="Model name (base or large) - default: large for H100")
    parser.add_argument("--no-kfold", action="store_true", 
                       help="Disable k-fold CV (use single train/val split)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--use-mean-pooling", action="store_true",
                       help="Use mean pooling instead of [CLS] token")
    
    args = parser.parse_args()
    
    # Create output directory with proper permissions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except PermissionError:
        # Try to fix permissions on parent directory
        parent_dir = os.path.dirname(output_dir)
        if os.path.exists(parent_dir):
            try:
                os.chmod(parent_dir, 0o755)
                os.makedirs(output_dir, exist_ok=True)
            except:
                # Fall back to current directory
                output_dir = f"run_{timestamp}"
                os.makedirs(output_dir, exist_ok=True)
                logger.warning(f"Using current directory for output: {output_dir}")
        else:
            # Create parent directory first
            os.makedirs(parent_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Improved DeBERTa Price Prediction with Feature Engineering")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"K-Fold CV: {not args.no_kfold} ({args.n_folds} folds)")
    logger.info(f"Mean Pooling: {args.use_mean_pooling}")
    logger.info(f"Output: {output_dir}")
    
    # Load data
    logger.info("\nLoading data...")
    train_path = os.path.join(args.data_dir, "train.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    
    train_df_raw = pd.read_csv(train_path)
    test_df_raw = pd.read_csv(test_path)
    
    logger.info(f"Raw training samples: {len(train_df_raw)}")
    logger.info(f"Raw test samples: {len(test_df_raw)}")
    
    # Preprocess data with feature extraction
    logger.info("\nPreprocessing data with feature extraction...")
    preprocessor = DataPreprocessor(
        extract_features=True,  # KEY: Enable feature extraction
        clean_text=True,
        remove_html=True,
        normalize_unicode=True,
    )
    
    # Preprocess training data
    train_df = preprocessor.preprocess_training_data(train_df_raw, fit_scaler=True)
    
    # Get unit columns from training data
    unit_columns = [col for col in train_df.columns if col.startswith('unit_')]
    
    # Preprocess test data
    test_df = preprocessor.preprocess_test_data(test_df_raw, unit_columns=unit_columns)
    
    logger.info(f"Preprocessed training samples: {len(train_df)}")
    logger.info(f"Preprocessed test samples: {len(test_df)}")
    
    # Get numerical feature columns
    numerical_feature_cols = get_numerical_feature_columns(train_df)
    logger.info(f"\nExtracted {len(numerical_feature_cols)} numerical features")
    
    # Train model
    if args.no_kfold:
        # Single train/val split
        logger.info("\nTraining with single train/val split...")
        
        trainer = PricePredictionTrainer(
            model_name=args.model_name,
            batch_size=64,  # Increased for H100
            gradient_accumulation_steps=1,  # Not needed with H100
            num_epochs=3,  # 3 epochs per fold is optimal
            learning_rate_encoder=1e-5,
            learning_rate_head=1e-3,
            num_numerical_features=len(numerical_feature_cols),
            use_mean_pooling=args.use_mean_pooling,
        )
        
        best_model_path = trainer.train(train_df, numerical_feature_cols, output_dir)
        model_paths = [best_model_path]
    else:
        # K-fold cross-validation
        model_paths, fold_results = train_with_kfold(
            train_df,
            model_name=args.model_name,
            n_folds=args.n_folds,
            use_mean_pooling=args.use_mean_pooling,
            output_dir=output_dir,
        )
    
    # Make predictions on test set
    logger.info("\nMaking predictions on test set...")
    
    if len(model_paths) > 1:
        # Ensemble predictions from k-fold models
        test_predictions = predict_with_ensemble(
            test_df,
            model_paths,
            numerical_feature_cols,
            model_name=args.model_name,
            use_mean_pooling=args.use_mean_pooling,
        )
    else:
        # Single model prediction
        trainer = PricePredictionTrainer(
            model_name=args.model_name,
            batch_size=64,  # Increased for H100
            num_numerical_features=len(numerical_feature_cols),
            use_mean_pooling=args.use_mean_pooling,
        )
        
        test_texts = test_df["catalog_content"].tolist()
        test_num_feats = test_df[numerical_feature_cols].values if numerical_feature_cols else None
        test_predictions = trainer.predict(test_texts, test_num_feats, model_paths[0])
    
    # Create submission
    submission = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "price": test_predictions
    })
    
    submission_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    
    logger.info(f"\n✅ Predictions saved to {submission_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    
    # Save configuration
    config = {
        'model_name': args.model_name,
        'use_kfold': not args.no_kfold,
        'n_folds': args.n_folds if not args.no_kfold else 1,
        'use_mean_pooling': args.use_mean_pooling,
        'num_numerical_features': len(numerical_feature_cols),
        'numerical_feature_cols': numerical_feature_cols,
        'training_samples': len(train_df),
        'test_samples': len(test_df),
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\n🎉 Training and prediction completed!")
    logger.info(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
