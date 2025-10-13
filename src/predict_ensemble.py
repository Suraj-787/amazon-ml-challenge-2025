#!/usr/bin/env python3
"""
Generate predictions using ensemble of trained models
Usage: python src/predict_ensemble.py --model-dir output/improved/run_20251013_145231
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import DataPreprocessor
from src.deberta_price_predictor import PricePredictionTrainer

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_numerical_feature_columns(df: pd.DataFrame) -> list:
    """Get list of numerical feature columns from preprocessed dataframe"""
    feature_cols = []
    
    if 'value' in df.columns:
        feature_cols.append('value')
    if 'pack_count' in df.columns:
        feature_cols.append('pack_count')
    
    unit_cols = [col for col in df.columns if col.startswith('unit_')]
    feature_cols.extend(unit_cols)
    
    return feature_cols


def main():
    parser = argparse.ArgumentParser(description="Generate ensemble predictions")
    parser.add_argument("--model-dir", required=True, help="Directory containing fold models")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large", help="Model name")
    parser.add_argument("--n-folds", type=int, default=3, help="Number of folds to use")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Ensemble Prediction Script")
    logger.info("="*60)
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Using {args.n_folds} folds")
    
    # Load test data
    logger.info("\nLoading test data...")
    test_path = os.path.join(args.data_dir, "test.csv")
    test_df_raw = pd.read_csv(test_path)
    logger.info(f"Test samples: {len(test_df_raw)}")
    
    # Load training data for preprocessing fit
    logger.info("Loading training data for preprocessing...")
    train_path = os.path.join(args.data_dir, "train.csv")
    train_df_raw = pd.read_csv(train_path)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor(
        extract_features=True,
        clean_text=True,
        remove_html=True,
        normalize_unicode=True,
    )
    
    # Fit on training data
    train_df = preprocessor.preprocess_training_data(train_df_raw, fit_scaler=True)
    unit_columns = [col for col in train_df.columns if col.startswith('unit_')]
    
    # Preprocess test data
    test_df = preprocessor.preprocess_test_data(test_df_raw, unit_columns=unit_columns)
    
    # Get numerical features
    numerical_feature_cols = get_numerical_feature_columns(test_df)
    num_numerical_features = len(numerical_feature_cols)
    logger.info(f"Using {num_numerical_features} numerical features")
    
    # Prepare test data
    test_texts = test_df["catalog_content"].tolist()
    test_num_feats = test_df[numerical_feature_cols].values if numerical_feature_cols else None
    
    # Make predictions with each fold
    all_predictions = []
    
    for fold in range(1, args.n_folds + 1):
        model_path = os.path.join(args.model_dir, f"fold_{fold}", "best_model.pt")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}, skipping...")
            continue
        
        logger.info(f"\nPredicting with Fold {fold}...")
        logger.info(f"Model: {model_path}")
        
        # Initialize trainer
        trainer = PricePredictionTrainer(
            model_name=args.model_name,
            batch_size=64,
            num_numerical_features=num_numerical_features,
            use_mean_pooling=False,
        )
        
        # Make predictions
        predictions = trainer.predict(test_texts, test_num_feats, model_path)
        all_predictions.append(predictions)
        
        logger.info(f"Fold {fold} predictions: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
    
    # Ensemble predictions (average)
    logger.info(f"\nAveraging predictions from {len(all_predictions)} models...")
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    logger.info(f"Ensemble predictions: min={ensemble_predictions.min():.2f}, max={ensemble_predictions.max():.2f}, mean={ensemble_predictions.mean():.2f}")
    
    # Create submission
    submission = pd.DataFrame({
        "sample_id": test_df_raw["sample_id"],
        "price": ensemble_predictions
    })
    
    # Save submission
    output_path = os.path.join(args.model_dir, "submission.csv")
    submission.to_csv(output_path, index=False)
    
    logger.info(f"\n✅ Submission saved to: {output_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Sample predictions:")
    logger.info(submission.head(10))
    
    # Validate submission format
    logger.info("\nValidating submission format...")
    assert list(submission.columns) == ["sample_id", "price"], "Invalid columns"
    assert len(submission) == len(test_df_raw), "Row count mismatch"
    assert submission["price"].notna().all(), "Missing predictions"
    logger.info("✅ Submission format validated!")


if __name__ == "__main__":
    main()
