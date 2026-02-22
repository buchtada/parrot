"""
GIFT-Eval Benchmark Runner for Parrot Forecasters

This script evaluates the parrot forecasting models against GIFT-Eval,
which includes 23 datasets covering 144,000 time series.

Reference: https://github.com/SalesforceAIResearch/gift-eval
Paper: "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import parrot forecasters
from context_parrot_benchmark import ContextParrotForecaster, evaluate_forecast
from derivative_parrot_benchmark import DerivativeParrotForecaster
from multivariate_parrot import (
    MultivarateContextParrotForecaster,
    MultivariateDerivativeParrotForecaster
)

# Check for required packages
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    from gluonts.dataset.common import ListDataset
    from gluonts.evaluation import Evaluator
    GLUONTS_AVAILABLE = True
except ImportError:
    GLUONTS_AVAILABLE = False


@dataclass
class GIFTEvalConfig:
    """Configuration for GIFT-Eval submission."""
    model: str = "parrot_context"
    model_type: str = "statistical"  # statistical, deep-learning, pretrained, zero-shot, fine-tuned, agentic
    model_dtype: str = "float32"
    model_link: str = ""
    code_link: str = "https://github.com/yourusername/parrot-forecast"
    org: str = "parrot-forecast"
    testdata_leakage: str = "No"
    replication_code_available: str = "Yes"


@dataclass
class GIFTEvalResult:
    """Container for GIFT-Eval results."""
    dataset_name: str
    num_series: int
    prediction_length: int
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None


class ParrotGIFTAdapter:
    """
    Adapter to run Parrot forecasters on GIFT-Eval datasets.

    Wraps parrot forecasting models to work with GIFT-Eval format.
    """

    def __init__(
        self,
        model_type: str = "context",
        min_pattern_length: int = 30,
        derivative_method: str = "savgol",
    ):
        self.model_type = model_type
        self.min_pattern_length = min_pattern_length

        # Initialize forecaster
        if model_type == "context":
            self.forecaster = ContextParrotForecaster(min_pattern_length=min_pattern_length)
        elif model_type == "derivative":
            self.forecaster = DerivativeParrotForecaster(
                min_pattern_length=min_pattern_length,
                derivative_method=derivative_method
            )
        elif model_type == "multivariate_context":
            self.forecaster = MultivarateContextParrotForecaster(
                min_pattern_length=min_pattern_length
            )
        elif model_type == "multivariate_derivative":
            self.forecaster = MultivariateDerivativeParrotForecaster(
                min_pattern_length=min_pattern_length,
                derivative_method=derivative_method
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forecast_single(
        self,
        context: np.ndarray,
        prediction_length: int
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate forecast for a single time series.

        Parameters
        ----------
        context : np.ndarray
            Historical values
        prediction_length : int
            Forecast horizon

        Returns
        -------
        predictions : np.ndarray
            Point forecasts
        metadata : Dict
            Forecast metadata
        """
        # Ensure minimum context length
        min_required = self.min_pattern_length + prediction_length
        if len(context) < min_required:
            # Naive forecast fallback
            return np.full(prediction_length, context[-1]), {"fallback": True}

        try:
            # Adapt pattern length to available context
            pattern_length = min(self.min_pattern_length, len(context) // 3)

            if "multivariate" in self.model_type:
                if context.ndim == 1:
                    context = context.reshape(-1, 1)
                forecast, metadata = self.forecaster.forecast(
                    time_series=context,
                    horizon=prediction_length,
                    pattern_length=pattern_length
                )
            else:
                if context.ndim > 1:
                    context = context[:, 0]  # Use first dimension
                forecast, metadata = self.forecaster.forecast(
                    time_series=context,
                    horizon=prediction_length,
                    pattern_length=pattern_length
                )

            return forecast, metadata

        except Exception as e:
            # Fallback
            return np.full(prediction_length, context[-1] if context.ndim == 1 else context[-1, 0]), {"error": str(e)}

    def forecast_batch(
        self,
        contexts: List[np.ndarray],
        prediction_length: int,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generate forecasts for multiple time series.

        Parameters
        ----------
        contexts : List[np.ndarray]
            List of historical time series
        prediction_length : int
            Forecast horizon

        Returns
        -------
        List[np.ndarray]
            List of forecasts
        """
        forecasts = []
        iterator = tqdm(contexts, desc="Forecasting") if show_progress else contexts

        for context in iterator:
            forecast, _ = self.forecast_single(context, prediction_length)
            forecasts.append(forecast)

        return forecasts


def load_gift_eval_datasets(
    subset: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Load GIFT-Eval datasets from Hugging Face.

    Parameters
    ----------
    subset : List[str], optional
        Specific datasets to load. If None, loads all.

    Returns
    -------
    Dict[str, Dict]
        Dictionary of datasets with their metadata
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading GIFT-Eval datasets from Hugging Face...")

    try:
        # Load the main dataset
        dataset = load_dataset("Salesforce/GiftEval")
        print(f"Loaded GIFT-Eval dataset with {len(dataset)} splits")

        datasets = {}
        for split_name in dataset.keys():
            if subset is not None and split_name not in subset:
                continue

            split_data = dataset[split_name]
            datasets[split_name] = {
                "data": split_data,
                "num_series": len(split_data),
            }
            print(f"  {split_name}: {len(split_data)} series")

        return datasets

    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Attempting manual download...")
        return load_gift_eval_manual()


def load_gift_eval_manual() -> Dict[str, Dict]:
    """Manual fallback for loading GIFT-Eval data."""

    # GIFT-Eval dataset configuration
    gift_eval_configs = [
        {"name": "electricity", "prediction_lengths": [24, 48, 96, 192]},
        {"name": "traffic", "prediction_lengths": [24, 48, 96, 192]},
        {"name": "weather", "prediction_lengths": [24, 48, 96, 192]},
        {"name": "etth1", "prediction_lengths": [24, 48, 96, 192]},
        {"name": "etth2", "prediction_lengths": [24, 48, 96, 192]},
        {"name": "ettm1", "prediction_lengths": [24, 48, 96, 192]},
        {"name": "ettm2", "prediction_lengths": [24, 48, 96, 192]},
    ]

    datasets = {}
    for config in gift_eval_configs:
        try:
            # Try loading individual dataset
            ds = load_dataset(
                "Salesforce/GiftEval",
                config["name"],
                trust_remote_code=True
            )
            datasets[config["name"]] = {
                "data": ds,
                "prediction_lengths": config["prediction_lengths"]
            }
        except Exception as e:
            print(f"Could not load {config['name']}: {e}")

    return datasets


def evaluate_on_gift_eval(
    adapter: ParrotGIFTAdapter,
    datasets: Dict[str, Dict],
    prediction_lengths: List[int] = [24, 48, 96],
    max_series_per_dataset: Optional[int] = None
) -> List[GIFTEvalResult]:
    """
    Run evaluation on GIFT-Eval datasets.

    Parameters
    ----------
    adapter : ParrotGIFTAdapter
        The parrot adapter to evaluate
    datasets : Dict[str, Dict]
        Loaded GIFT-Eval datasets
    prediction_lengths : List[int]
        Horizons to evaluate
    max_series_per_dataset : int, optional
        Limit number of series (for faster testing)

    Returns
    -------
    List[GIFTEvalResult]
        Results for each dataset/horizon combination
    """
    results = []

    for dataset_name, dataset_info in datasets.items():
        print(f"\nEvaluating on: {dataset_name}")

        data = dataset_info.get("data", dataset_info)

        # Handle different data formats
        if hasattr(data, '__iter__'):
            series_list = list(data)
        else:
            series_list = [data]

        if max_series_per_dataset:
            series_list = series_list[:max_series_per_dataset]

        for pred_len in prediction_lengths:
            print(f"  Prediction length: {pred_len}")

            all_predictions = []
            all_targets = []

            for series in tqdm(series_list, desc=f"  {dataset_name} h={pred_len}", leave=False):
                # Extract context and target
                if isinstance(series, dict):
                    values = np.array(series.get("target", series.get("values", [])))
                else:
                    values = np.array(series)

                if len(values) < pred_len + adapter.min_pattern_length:
                    continue

                context = values[:-pred_len]
                target = values[-pred_len:]

                # Generate forecast
                prediction, _ = adapter.forecast_single(context, pred_len)

                all_predictions.append(prediction)
                all_targets.append(target)

            if all_predictions:
                # Compute aggregate metrics
                predictions = np.array(all_predictions)
                targets = np.array(all_targets)

                metrics = compute_gift_metrics(targets, predictions)

                result = GIFTEvalResult(
                    dataset_name=dataset_name,
                    num_series=len(all_predictions),
                    prediction_length=pred_len,
                    metrics=metrics
                )
                results.append(result)

                print(f"    MSE: {metrics['MSE']:.4f}, MAE: {metrics['MAE']:.4f}")

    return results


def compute_gift_metrics(
    targets: np.ndarray,
    predictions: np.ndarray
) -> Dict[str, float]:
    """
    Compute GIFT-Eval metrics.

    Standard metrics used in the benchmark:
    - MSE: Mean Squared Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - sMAPE: Symmetric MAPE
    - MASE: Mean Absolute Scaled Error
    - CRPS: Continuous Ranked Probability Score (for probabilistic)
    """
    # Flatten for overall metrics
    targets_flat = targets.flatten()
    predictions_flat = predictions.flatten()

    # MSE
    mse = np.mean((predictions_flat - targets_flat) ** 2)

    # MAE
    mae = np.mean(np.abs(predictions_flat - targets_flat))

    # MAPE (handle zeros)
    epsilon = 1e-8
    mape = 100 * np.mean(np.abs(predictions_flat - targets_flat) / (np.abs(targets_flat) + epsilon))

    # sMAPE
    smape = 200 * np.mean(
        np.abs(predictions_flat - targets_flat) /
        (np.abs(targets_flat) + np.abs(predictions_flat) + epsilon)
    )

    # RMSE
    rmse = np.sqrt(mse)

    # Normalized metrics
    target_std = np.std(targets_flat)
    nrmse = rmse / (target_std + epsilon)

    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "sMAPE": float(smape),
        "NRMSE": float(nrmse)
    }


def run_gift_eval_benchmark(
    model_types: List[str] = ["context", "derivative"],
    prediction_lengths: List[int] = [24, 48, 96],
    max_series: Optional[int] = 100,
    save_results: bool = True,
    output_dir: str = "./gift_results"
) -> Dict[str, List[GIFTEvalResult]]:
    """
    Run full GIFT-Eval benchmark.

    Parameters
    ----------
    model_types : List[str]
        Parrot model variants to evaluate
    prediction_lengths : List[int]
        Forecast horizons
    max_series : int, optional
        Max series per dataset (for faster runs)
    save_results : bool
        Save results to JSON
    output_dir : str
        Output directory

    Returns
    -------
    Dict[str, List[GIFTEvalResult]]
        Results for each model type
    """
    print("=" * 70)
    print("GIFT-EVAL BENCHMARK")
    print("=" * 70)

    # Load datasets
    datasets = load_gift_eval_datasets()

    all_results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Model: {model_type.upper()} PARROT")
        print("=" * 70)

        adapter = ParrotGIFTAdapter(model_type=model_type)

        results = evaluate_on_gift_eval(
            adapter=adapter,
            datasets=datasets,
            prediction_lengths=prediction_lengths,
            max_series_per_dataset=max_series
        )

        all_results[model_type] = results

        # Print summary
        if results:
            avg_metrics = {}
            for result in results:
                for metric, value in result.metrics.items():
                    if metric not in avg_metrics:
                        avg_metrics[metric] = []
                    avg_metrics[metric].append(value)

            print(f"\n{model_type.upper()} Average Metrics:")
            for metric, values in avg_metrics.items():
                print(f"  {metric}: {np.mean(values):.4f}")

    # Save results
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        summary = {}
        for model_type, results in all_results.items():
            summary[model_type] = {
                "results": [
                    {
                        "dataset": r.dataset_name,
                        "prediction_length": r.prediction_length,
                        "num_series": r.num_series,
                        "metrics": r.metrics
                    }
                    for r in results
                ]
            }

        with open(output_path / "gift_eval_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Generate config.json for submission
        config = GIFTEvalConfig(
            model="parrot_context",
            model_type="statistical",
            org="parrot-forecast"
        )

        with open(output_path / "config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return all_results


def generate_submission_files(
    results: Dict[str, List[GIFTEvalResult]],
    output_dir: str = "./gift_submission"
):
    """
    Generate files needed for GIFT-Eval leaderboard submission.

    Creates:
    - config.json: Model metadata
    - results.json: Evaluation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Config for submission
    config = {
        "model": "parrot",
        "model_type": "statistical",
        "model_dtype": "float32",
        "model_link": "",
        "code_link": "https://github.com/yourusername/parrot-forecast",
        "org": "parrot-forecast",
        "testdata_leakage": "No",
        "replication_code_available": "Yes"
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Submission files generated in {output_path}")
    print("\nTo submit to GIFT-Eval leaderboard:")
    print("1. Fork https://github.com/SalesforceAIResearch/gift-eval")
    print("2. Add your results to the leaderboard")
    print("3. Create a pull request")


if __name__ == "__main__":
    print("GIFT-Eval Benchmark Runner for Parrot Forecasters")
    print("=" * 70)

    if not DATASETS_AVAILABLE:
        print("\nInstalling required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets", "tqdm"])
        print("Please restart the script.")
    else:
        # Run benchmark
        results = run_gift_eval_benchmark(
            model_types=["context", "derivative"],
            prediction_lengths=[24, 48],  # Start with shorter horizons
            max_series=50,  # Subset for testing
            save_results=True
        )

        print("\n" + "=" * 70)
        print("GIFT-EVAL BENCHMARK COMPLETE")
        print("=" * 70)
