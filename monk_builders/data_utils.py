"""
Data utilities for preprocessing and validation
Supports the DeBERTa price prediction pipeline
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from typing import List, Tuple, Optional, Dict
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing utilities for product catalog data"""

    def __init__(
        self,
        min_price: float = 0.01,
        max_price: float = 10000.0,
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
            # Keep: letters, numbers, spaces, basic punctuation
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

        # Remove rows with missing catalog_content
        before_content_filter = len(df)
        df = df.dropna(subset=["catalog_content"])
        after_content_filter = len(df)
        if before_content_filter != after_content_filter:
            logger.warning(
                f"Removed {before_content_filter - after_content_filter} rows with missing catalog_content"
            )

        # Remove rows with missing prices
        before_price_filter = len(df)
        df = df.dropna(subset=["price"])
        after_price_filter = len(df)
        if before_price_filter != after_price_filter:
            logger.warning(
                f"Removed {before_price_filter - after_price_filter} rows with missing prices"
            )

        # Clean catalog content
        logger.info("Cleaning catalog content...")
        df["catalog_content"] = df["catalog_content"].apply(self.clean_catalog_content)

        # Remove empty content after cleaning
        before_empty_filter = len(df)
        df = df[df["catalog_content"].str.len() > 0]
        after_empty_filter = len(df)
        if before_empty_filter != after_empty_filter:
            logger.warning(
                f"Removed {before_empty_filter - after_empty_filter} rows with empty content after cleaning"
            )

        # Filter valid prices
        before_price_validation = len(df)
        valid_price_mask = df["price"].apply(self.validate_price)
        df = df[valid_price_mask]
        after_price_validation = len(df)
        if before_price_validation != after_price_validation:
            logger.warning(
                f"Removed {before_price_validation - after_price_validation} rows with invalid prices"
            )

        # Remove duplicates based on catalog_content
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["catalog_content"], keep="first")
        after_dedup = len(df)
        if before_dedup != after_dedup:
            logger.warning(f"Removed {before_dedup - after_dedup} duplicate entries")

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

        original_count = len(df)
        logger.info(f"Original test samples: {original_count}")

        # Check required columns
        required_cols = ["catalog_content", "sample_id"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Handle missing catalog_content (fill with empty string)
        missing_content_count = df["catalog_content"].isna().sum()
        if missing_content_count > 0:
            logger.warning(
                f"Found {missing_content_count} rows with missing catalog_content, filling with empty string"
            )
            df["catalog_content"] = df["catalog_content"].fillna("")

        # Clean catalog content
        logger.info("Cleaning catalog content...")
        df["catalog_content"] = df["catalog_content"].apply(self.clean_catalog_content)

        # For test data, we keep all rows even if content is empty
        # The model should handle empty content gracefully
        empty_content_count = (df["catalog_content"].str.len() == 0).sum()
        if empty_content_count > 0:
            logger.warning(
                f"Found {empty_content_count} rows with empty content after cleaning"
            )

        final_count = len(df)
        logger.info(f"Final test samples: {final_count}")

        return df.reset_index(drop=True)


class DataValidator:
    """Data validation utilities"""

    @staticmethod
    def validate_training_data(df: pd.DataFrame) -> Dict[str, any]:
        """Validate training data and return statistics"""
        stats = {}

        # Basic info
        stats["total_samples"] = len(df)
        stats["columns"] = list(df.columns)

        # Required columns check
        required_cols = ["catalog_content", "price", "sample_id"]
        stats["missing_required_columns"] = [
            col for col in required_cols if col not in df.columns
        ]

        # Missing values
        stats["missing_values"] = df.isnull().sum().to_dict()

        # Duplicate check
        stats["duplicate_sample_ids"] = df["sample_id"].duplicated().sum()
        stats["duplicate_content"] = df["catalog_content"].duplicated().sum()

        if "price" in df.columns:
            # Price statistics
            price_stats = df["price"].describe().to_dict()
            stats["price_statistics"] = price_stats
            stats["negative_prices"] = (df["price"] < 0).sum()
            stats["zero_prices"] = (df["price"] == 0).sum()
            stats["very_high_prices"] = (df["price"] > 10000).sum()

        if "catalog_content" in df.columns:
            # Text statistics
            text_lengths = df["catalog_content"].str.len()
            stats["text_length_statistics"] = text_lengths.describe().to_dict()
            stats["empty_content"] = (text_lengths == 0).sum()
            stats["very_short_content"] = (text_lengths < 10).sum()
            stats["very_long_content"] = (text_lengths > 1000).sum()

        return stats

    @staticmethod
    def validate_test_data(df: pd.DataFrame) -> Dict[str, any]:
        """Validate test data and return statistics"""
        stats = {}

        # Basic info
        stats["total_samples"] = len(df)
        stats["columns"] = list(df.columns)

        # Required columns check
        required_cols = ["catalog_content", "sample_id"]
        stats["missing_required_columns"] = [
            col for col in required_cols if col not in df.columns
        ]

        # Missing values
        stats["missing_values"] = df.isnull().sum().to_dict()

        # Duplicate check
        stats["duplicate_sample_ids"] = df["sample_id"].duplicated().sum()

        if "catalog_content" in df.columns:
            # Text statistics
            text_lengths = df["catalog_content"].str.len()
            stats["text_length_statistics"] = text_lengths.describe().to_dict()
            stats["empty_content"] = (text_lengths == 0).sum()
            stats["very_short_content"] = (text_lengths < 10).sum()
            stats["very_long_content"] = (text_lengths > 1000).sum()

        return stats

    @staticmethod
    def print_validation_report(stats: Dict[str, any], data_type: str = "training"):
        """Print a formatted validation report"""
        print(f"\n📊 {data_type.title()} Data Validation Report")
        print("=" * 50)

        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Columns: {stats['columns']}")

        if stats["missing_required_columns"]:
            print(f"❌ Missing Required Columns: {stats['missing_required_columns']}")
        else:
            print("✅ All required columns present")

        # Missing values
        missing_values = stats["missing_values"]
        total_missing = sum(missing_values.values())
        if total_missing > 0:
            print(f"\n⚠️  Missing Values:")
            for col, count in missing_values.items():
                if count > 0:
                    print(
                        f"  - {col}: {count:,} ({count / stats['total_samples'] * 100:.2f}%)"
                    )
        else:
            print("\n✅ No missing values")

        # Duplicates
        if "duplicate_sample_ids" in stats:
            if stats["duplicate_sample_ids"] > 0:
                print(f"\n⚠️  Duplicate sample IDs: {stats['duplicate_sample_ids']:,}")
            else:
                print("\n✅ No duplicate sample IDs")

        # Price statistics (training only)
        if "price_statistics" in stats:
            price_stats = stats["price_statistics"]
            print(f"\n💰 Price Statistics:")
            print(f"  - Mean: ${price_stats['mean']:.2f}")
            print(f"  - Median: ${price_stats['50%']:.2f}")
            print(f"  - Min: ${price_stats['min']:.2f}")
            print(f"  - Max: ${price_stats['max']:.2f}")
            print(f"  - Std: ${price_stats['std']:.2f}")

            # Price issues
            issues = []
            if stats["negative_prices"] > 0:
                issues.append(f"Negative prices: {stats['negative_prices']:,}")
            if stats["zero_prices"] > 0:
                issues.append(f"Zero prices: {stats['zero_prices']:,}")
            if stats["very_high_prices"] > 0:
                issues.append(
                    f"Very high prices (>$10k): {stats['very_high_prices']:,}"
                )

            if issues:
                print(f"  ⚠️  Price Issues: {', '.join(issues)}")

        # Text statistics
        if "text_length_statistics" in stats:
            text_stats = stats["text_length_statistics"]
            print(f"\n📝 Text Length Statistics:")
            print(f"  - Mean: {text_stats['mean']:.0f} chars")
            print(f"  - Median: {text_stats['50%']:.0f} chars")
            print(f"  - Min: {text_stats['min']:.0f} chars")
            print(f"  - Max: {text_stats['max']:.0f} chars")

            # Text issues
            text_issues = []
            if stats["empty_content"] > 0:
                text_issues.append(f"Empty content: {stats['empty_content']:,}")
            if stats["very_short_content"] > 0:
                text_issues.append(
                    f"Very short (<10 chars): {stats['very_short_content']:,}"
                )
            if stats["very_long_content"] > 0:
                text_issues.append(
                    f"Very long (>1000 chars): {stats['very_long_content']:,}"
                )

            if text_issues:
                print(f"  ⚠️  Text Issues: {', '.join(text_issues)}")

        print("=" * 50)


def load_and_preprocess_data(
    train_path: str, test_path: str, validate: bool = True, **preprocessor_kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess training and test data

    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
        validate: Whether to run validation and print reports
        **preprocessor_kwargs: Arguments for DataPreprocessor

    Returns:
        Tuple of (preprocessed_train_df, preprocessed_test_df)
    """
    logger.info("Loading raw data...")

    # Load raw data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(
        f"Loaded {len(train_df)} training samples and {len(test_df)} test samples"
    )

    # Validate raw data
    if validate:
        print("\n🔍 Raw Data Validation")
        train_stats = DataValidator.validate_training_data(train_df)
        DataValidator.print_validation_report(train_stats, "training")

        test_stats = DataValidator.validate_test_data(test_df)
        DataValidator.print_validation_report(test_stats, "test")

    # Preprocess data
    preprocessor = DataPreprocessor(**preprocessor_kwargs)

    train_df_clean = preprocessor.preprocess_training_data(train_df)
    test_df_clean = preprocessor.preprocess_test_data(test_df)

    # Validate processed data
    if validate:
        print("\n🧹 Processed Data Validation")
        train_stats_clean = DataValidator.validate_training_data(train_df_clean)
        DataValidator.print_validation_report(train_stats_clean, "processed training")

        test_stats_clean = DataValidator.validate_test_data(test_df_clean)
        DataValidator.print_validation_report(test_stats_clean, "processed test")

    return train_df_clean, test_df_clean


def split_training_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split training data into train and validation sets

    Args:
        df: Training dataframe
        test_size: Fraction for validation set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df)
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=None,  # Could add price-based stratification if needed
    )

    logger.info(f"Split data: {len(train_df)} training, {len(val_df)} validation")

    return train_df, val_df


if __name__ == "__main__":
    # Example usage
    train_df, test_df = load_and_preprocess_data(
        "data/train.csv", "data/test.csv", validate=True
    )

    print(f"\nFinal preprocessed data:")
    print(f"Training: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")
