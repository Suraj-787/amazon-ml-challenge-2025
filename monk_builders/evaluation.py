"""
Evaluation utilities for price prediction model
Implements SMAPE and other relevant metrics for the challenge
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def calculate_smape(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)

    This is the main evaluation metric for the challenge.
    SMAPE = 100 * mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        SMAPE value as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure no negative values (prices should be positive)
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon

    smape = 100 * np.mean(numerator / denominator)
    return smape


def calculate_mape(
    y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE)

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return float("inf")

    mape = 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mape


def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE)

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        WMAPE value as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    return wmape


def calculate_log_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics in log space (useful for price prediction)

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary of log-space metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Add 1 to handle zero values
    log_true = np.log(y_true + 1)
    log_pred = np.log(y_pred + 1)

    metrics = {
        "log_mae": mean_absolute_error(log_true, log_pred),
        "log_mse": mean_squared_error(log_true, log_pred),
        "log_rmse": np.sqrt(mean_squared_error(log_true, log_pred)),
        "log_r2": r2_score(log_true, log_pred),
    }

    return metrics


def calculate_price_range_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    price_ranges: List[Tuple[float, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different price ranges

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        price_ranges: List of (min, max) tuples for price ranges

    Returns:
        Dictionary of metrics by price range
    """
    if price_ranges is None:
        price_ranges = [
            (0, 10),  # Low price
            (10, 50),  # Medium price
            (50, 200),  # High price
            (200, float("inf")),  # Very high price
        ]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    range_metrics = {}

    for i, (min_price, max_price) in enumerate(price_ranges):
        range_name = (
            f"${min_price}-${max_price}"
            if max_price != float("inf")
            else f"${min_price}+"
        )

        # Create mask for this price range
        mask = (y_true >= min_price) & (y_true < max_price)

        if np.sum(mask) == 0:
            continue

        true_range = y_true[mask]
        pred_range = y_pred[mask]

        range_metrics[range_name] = {
            "count": len(true_range),
            "smape": calculate_smape(true_range, pred_range),
            "mape": calculate_mape(true_range, pred_range),
            "mae": mean_absolute_error(true_range, pred_range),
            "rmse": np.sqrt(mean_squared_error(true_range, pred_range)),
            "r2": r2_score(true_range, pred_range),
        }

    return range_metrics


def comprehensive_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, any]:
    """
    Perform comprehensive evaluation with all relevant metrics

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        Dictionary containing all evaluation results
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    results = {}

    # Basic info
    results["sample_count"] = len(y_true)
    results["price_range_true"] = (float(np.min(y_true)), float(np.max(y_true)))
    results["price_range_pred"] = (float(np.min(y_pred)), float(np.max(y_pred)))

    # Main metrics
    results["smape"] = calculate_smape(y_true, y_pred)
    results["mape"] = calculate_mape(y_true, y_pred)
    results["wmape"] = calculate_wmape(y_true, y_pred)

    # Standard regression metrics
    results["mae"] = mean_absolute_error(y_true, y_pred)
    results["mse"] = mean_squared_error(y_true, y_pred)
    results["rmse"] = np.sqrt(results["mse"])
    results["r2"] = r2_score(y_true, y_pred)

    # Log-space metrics
    results["log_metrics"] = calculate_log_metrics(y_true, y_pred)

    # Price range analysis
    results["price_range_metrics"] = calculate_price_range_metrics(y_true, y_pred)

    # Residual analysis
    residuals = y_pred - y_true
    results["residual_stats"] = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "median": float(np.median(residuals)),
        "q25": float(np.percentile(residuals, 25)),
        "q75": float(np.percentile(residuals, 75)),
    }

    return results


def print_evaluation_report(results: Dict[str, any]) -> None:
    """
    Print a formatted evaluation report

    Args:
        results: Results dictionary from comprehensive_evaluation
    """
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
    print(f"  MAPE: {results['mape']:.2f}%")
    print(f"  WMAPE: {results['wmape']:.2f}%")

    # Regression metrics
    print(f"\n📈 Regression Metrics:")
    print(f"  MAE: ${results['mae']:.2f}")
    print(f"  RMSE: ${results['rmse']:.2f}")
    print(f"  R²: {results['r2']:.4f}")

    # Log-space metrics
    log_metrics = results["log_metrics"]
    print(f"\n📊 Log-Space Metrics:")
    print(f"  Log MAE: {log_metrics['log_mae']:.4f}")
    print(f"  Log RMSE: {log_metrics['log_rmse']:.4f}")
    print(f"  Log R²: {log_metrics['log_r2']:.4f}")

    # Price range analysis
    print(f"\n💰 Price Range Analysis:")
    for range_name, metrics in results["price_range_metrics"].items():
        print(f"  {range_name} ({metrics['count']:,} samples):")
        print(f"    SMAPE: {metrics['smape']:.2f}%")
        print(f"    MAE: ${metrics['mae']:.2f}")
        print(f"    R²: {metrics['r2']:.4f}")

    # Residual analysis
    residual_stats = results["residual_stats"]
    print(f"\n🔍 Residual Analysis:")
    print(f"  Mean Residual: ${residual_stats['mean']:.2f}")
    print(f"  Residual Std: ${residual_stats['std']:.2f}")
    print(f"  Residual Median: ${residual_stats['median']:.2f}")
    print(
        f"  Residual IQR: [${residual_stats['q25']:.2f}, ${residual_stats['q75']:.2f}]"
    )

    print("=" * 60)


def create_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str = None,
    show_plots: bool = True,
) -> None:
    """
    Create evaluation plots for model performance

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        output_dir: Directory to save plots (optional)
        show_plots: Whether to display plots
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Predicted vs True scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )

    ax1.set_xlabel("True Prices ($)")
    ax1.set_ylabel("Predicted Prices ($)")
    ax1.set_title("Predicted vs True Prices")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = y_pred - y_true
    ax2.scatter(y_true, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color="r", linestyle="--", lw=2)
    ax2.set_xlabel("True Prices ($)")
    ax2.set_ylabel("Residuals ($)")
    ax2.set_title("Residuals vs True Prices")
    ax2.grid(True, alpha=0.3)

    # 3. Log-scale scatter plot
    ax3 = axes[1, 0]
    ax3.scatter(np.log(y_true + 1), np.log(y_pred + 1), alpha=0.6, s=20)

    # Perfect prediction line in log space
    log_min = np.log(y_true.min() + 1)
    log_max = np.log(y_true.max() + 1)
    ax3.plot(
        [log_min, log_max], [log_min, log_max], "r--", lw=2, label="Perfect Prediction"
    )

    ax3.set_xlabel("Log(True Prices + 1)")
    ax3.set_ylabel("Log(Predicted Prices + 1)")
    ax3.set_title("Log-Scale: Predicted vs True")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Error distribution
    ax4 = axes[1, 1]
    percentage_errors = (
        100 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)) * 2
    )
    ax4.hist(percentage_errors, bins=50, alpha=0.7, edgecolor="black")
    ax4.axvline(
        x=np.mean(percentage_errors),
        color="r",
        linestyle="--",
        lw=2,
        label=f"Mean: {np.mean(percentage_errors):.2f}%",
    )
    ax4.set_xlabel("Absolute Percentage Error (%)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of Absolute Percentage Errors")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        import os

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "evaluation_plots.png"),
            dpi=300,
            bbox_inches="tight",
        )
        logger.info(f"Evaluation plots saved to {output_dir}/evaluation_plots.png")

    if show_plots:
        plt.show()
    else:
        plt.close()


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str = None,
    create_plots: bool = True,
) -> Dict[str, any]:
    """
    Complete model evaluation pipeline

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        output_dir: Directory to save results and plots
        create_plots: Whether to create evaluation plots

    Returns:
        Comprehensive evaluation results
    """
    logger.info("Starting comprehensive model evaluation...")

    # Calculate all metrics
    results = comprehensive_evaluation(y_true, y_pred)

    # Print report
    print_evaluation_report(results)

    # Create plots
    if create_plots:
        create_evaluation_plots(y_true, y_pred, output_dir, show_plots=False)

    # Save results to file
    if output_dir:
        import json
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            else:
                return obj

        results_json = convert_numpy_types(results)

        with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"Evaluation results saved to {output_dir}/evaluation_results.json")

    logger.info(f"✅ Evaluation completed! SMAPE: {results['smape']:.2f}%")
    return results


def compare_models(
    predictions_dict: Dict[str, np.ndarray], y_true: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models by their predictions

    Args:
        predictions_dict: Dictionary mapping model names to their predictions
        y_true: Ground truth values

    Returns:
        DataFrame with comparison metrics
    """
    comparison_results = []

    for model_name, y_pred in predictions_dict.items():
        results = comprehensive_evaluation(y_true, y_pred)

        comparison_results.append(
            {
                "Model": model_name,
                "SMAPE": results["smape"],
                "MAPE": results["mape"],
                "MAE": results["mae"],
                "RMSE": results["rmse"],
                "R²": results["r2"],
                "Log_R²": results["log_metrics"]["log_r2"],
            }
        )

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values("SMAPE", ascending=True)

    print("\n🏆 Model Comparison Results")
    print("=" * 80)
    print(comparison_df.round(4).to_string(index=False))
    print("=" * 80)

    return comparison_df


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate sample data
    n_samples = 1000
    y_true = np.random.lognormal(mean=2, sigma=1, size=n_samples)
    y_pred = y_true * (1 + np.random.normal(0, 0.15, size=n_samples))  # 15% noise
    y_pred = np.maximum(y_pred, 0.01)  # Ensure positive predictions

    # Evaluate model
    results = evaluate_model_predictions(y_true, y_pred, output_dir="sample_evaluation")

    print(f"\nSample evaluation completed with SMAPE: {results['smape']:.2f}%")
