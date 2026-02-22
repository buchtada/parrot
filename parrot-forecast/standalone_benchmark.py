#!/usr/bin/env python3
"""
Standalone Benchmark Evaluation for Parrot Forecasters

This script evaluates parrot forecasters against FEV-Bench and GIFT-Eval
datasets WITHOUT requiring the fev package (which needs Python 3.10+).

It directly loads datasets from Hugging Face and evaluates using
standard metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import parrot forecasters
from context_parrot_benchmark import ContextParrotForecaster, evaluate_forecast
from derivative_parrot_benchmark import DerivativeParrotForecaster


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    benchmark: str
    dataset: str
    model: str
    prediction_length: int
    num_series: int
    metrics: Dict[str, float]


class ParrotEvaluator:
    """Evaluator for parrot forecasters on benchmark datasets."""

    def __init__(self, model_type: str = "context", min_pattern_length: int = 30):
        self.model_type = model_type
        self.min_pattern_length = min_pattern_length

        if model_type == "context":
            self.forecaster = ContextParrotForecaster(min_pattern_length=min_pattern_length)
        elif model_type == "derivative":
            self.forecaster = DerivativeParrotForecaster(
                min_pattern_length=min_pattern_length,
                derivative_method='savgol'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forecast(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """Generate forecast for a single time series."""
        # Ensure minimum context length
        min_required = self.min_pattern_length + horizon
        if len(context) < min_required:
            return np.full(horizon, context[-1])

        try:
            pattern_length = min(self.min_pattern_length, len(context) // 3)
            forecast, _ = self.forecaster.forecast(
                time_series=context,
                horizon=horizon,
                pattern_length=pattern_length
            )
            return forecast
        except Exception as e:
            return np.full(horizon, context[-1])


def load_fev_benchmark_data():
    """
    Load FEV benchmark datasets from Hugging Face.

    Returns sample datasets for evaluation.
    """
    print("Loading FEV benchmark data...")

    try:
        from datasets import load_dataset

        # Load FEV datasets
        dataset = load_dataset("autogluon/fev_datasets", trust_remote_code=True)
        print(f"  Loaded FEV datasets: {list(dataset.keys())}")
        return dataset

    except Exception as e:
        print(f"  Could not load from HuggingFace: {e}")
        print("  Using synthetic test data instead...")
        return generate_synthetic_benchmark_data()


def load_gift_eval_data():
    """
    Load GIFT-Eval datasets from Hugging Face.

    Returns sample datasets for evaluation.
    """
    print("Loading GIFT-Eval data...")

    try:
        from datasets import load_dataset

        # Try loading GIFT-Eval
        dataset = load_dataset("Salesforce/GiftEval", trust_remote_code=True)
        print(f"  Loaded GIFT-Eval: {list(dataset.keys())}")
        return dataset

    except Exception as e:
        print(f"  Could not load from HuggingFace: {e}")
        print("  Using synthetic test data instead...")
        return generate_synthetic_benchmark_data()


def generate_synthetic_benchmark_data() -> Dict[str, List[Dict]]:
    """
    Generate synthetic benchmark data for testing.

    Creates time series similar to those in FEV-Bench/GIFT-Eval.
    """
    np.random.seed(42)

    datasets = {}

    # Dataset 1: Seasonal + trend (like electricity/traffic)
    n_series = 50
    series_length = 500
    synthetic_seasonal = []

    for i in range(n_series):
        t = np.arange(series_length)
        # Seasonal component
        period = 24 + np.random.randint(-4, 5)
        seasonal = np.sin(2 * np.pi * t / period) * (0.5 + 0.5 * np.random.rand())
        # Trend
        trend = 0.001 * t * (np.random.rand() - 0.5)
        # Noise
        noise = 0.1 * np.random.randn(series_length)
        # Combine
        values = 10 + seasonal + trend + noise
        synthetic_seasonal.append({
            "id": f"synthetic_seasonal_{i}",
            "target": values.tolist(),
            "prediction_length": 24
        })

    datasets["synthetic_seasonal"] = synthetic_seasonal

    # Dataset 2: Random walk (like financial data)
    synthetic_walk = []
    for i in range(n_series):
        steps = np.random.randn(series_length) * 0.05
        values = 100 + np.cumsum(steps)
        synthetic_walk.append({
            "id": f"synthetic_walk_{i}",
            "target": values.tolist(),
            "prediction_length": 48
        })

    datasets["synthetic_random_walk"] = synthetic_walk

    # Dataset 3: Chaotic-like (to test parrot's strength)
    synthetic_chaotic = []
    for i in range(n_series):
        t = np.linspace(0, 50, series_length)
        # Multiple frequencies (quasi-periodic)
        f1 = 0.1 + 0.05 * np.random.rand()
        f2 = 0.17 + 0.03 * np.random.rand()
        f3 = 0.23 + 0.02 * np.random.rand()
        values = (np.sin(2 * np.pi * f1 * t) +
                  0.5 * np.sin(2 * np.pi * f2 * t) +
                  0.3 * np.cos(2 * np.pi * f3 * t))
        values += 0.05 * np.random.randn(series_length)
        synthetic_chaotic.append({
            "id": f"synthetic_chaotic_{i}",
            "target": values.tolist(),
            "prediction_length": 24
        })

    datasets["synthetic_chaotic"] = synthetic_chaotic

    print(f"  Generated {len(datasets)} synthetic datasets")
    for name, data in datasets.items():
        print(f"    {name}: {len(data)} series")

    return datasets


def evaluate_on_dataset(
    evaluator: ParrotEvaluator,
    dataset: List[Dict],
    dataset_name: str,
    prediction_length: int = 24,
    max_series: Optional[int] = None
) -> BenchmarkResult:
    """
    Evaluate forecaster on a single dataset.

    Parameters
    ----------
    evaluator : ParrotEvaluator
        The forecaster to evaluate
    dataset : List[Dict]
        List of time series dictionaries
    dataset_name : str
        Name of the dataset
    prediction_length : int
        Forecast horizon
    max_series : int, optional
        Maximum number of series to evaluate

    Returns
    -------
    BenchmarkResult
        Evaluation results
    """
    if max_series:
        dataset = dataset[:max_series]

    all_predictions = []
    all_targets = []

    for item in dataset:
        # Extract target values
        if isinstance(item, dict):
            values = np.array(item.get("target", item.get("values", [])))
            pred_len = item.get("prediction_length", prediction_length)
        else:
            values = np.array(item)
            pred_len = prediction_length

        # Skip if too short
        if len(values) < pred_len + evaluator.min_pattern_length:
            continue

        # Split into context and target
        context = values[:-pred_len]
        target = values[-pred_len:]

        # Generate forecast
        prediction = evaluator.forecast(context, pred_len)

        all_predictions.append(prediction)
        all_targets.append(target)

    if not all_predictions:
        return BenchmarkResult(
            benchmark="unknown",
            dataset=dataset_name,
            model=evaluator.model_type,
            prediction_length=prediction_length,
            num_series=0,
            metrics={"error": "no valid series"}
        )

    # Compute aggregate metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Flatten for metrics
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    epsilon = 1e-8

    metrics = {
        "MSE": float(np.mean((pred_flat - target_flat) ** 2)),
        "MAE": float(np.mean(np.abs(pred_flat - target_flat))),
        "RMSE": float(np.sqrt(np.mean((pred_flat - target_flat) ** 2))),
        "sMAPE": float(200 * np.mean(
            np.abs(pred_flat - target_flat) /
            (np.abs(target_flat) + np.abs(pred_flat) + epsilon)
        )),
        "MAPE": float(100 * np.mean(
            np.abs(pred_flat - target_flat) / (np.abs(target_flat) + epsilon)
        ))
    }

    return BenchmarkResult(
        benchmark="fev/gift",
        dataset=dataset_name,
        model=evaluator.model_type,
        prediction_length=prediction_length,
        num_series=len(all_predictions),
        metrics=metrics
    )


def run_full_evaluation(
    model_types: List[str] = ["context", "derivative"],
    prediction_lengths: List[int] = [24, 48],
    max_series: int = 50,
    save_results: bool = True
) -> Dict[str, List[BenchmarkResult]]:
    """
    Run full benchmark evaluation.

    Parameters
    ----------
    model_types : List[str]
        Model variants to test
    prediction_lengths : List[int]
        Forecast horizons to evaluate
    max_series : int
        Max series per dataset
    save_results : bool
        Save results to JSON

    Returns
    -------
    Dict[str, List[BenchmarkResult]]
        Results for each model type
    """
    print("=" * 70)
    print("PARROT FORECASTER BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {model_types}")
    print(f"Prediction lengths: {prediction_lengths}")
    print(f"Max series per dataset: {max_series}")
    print()

    # Load datasets
    print("Loading benchmark datasets...")

    # Try to load real datasets, fall back to synthetic
    try:
        fev_data = load_fev_benchmark_data()
    except Exception as e:
        print(f"FEV load failed: {e}")
        fev_data = None

    try:
        gift_data = load_gift_eval_data()
    except Exception as e:
        print(f"GIFT load failed: {e}")
        gift_data = None

    # If both fail, use synthetic
    if fev_data is None and gift_data is None:
        print("\nUsing synthetic benchmark data...")
        datasets = generate_synthetic_benchmark_data()
    else:
        datasets = {}
        if fev_data is not None:
            for split in fev_data.keys():
                datasets[f"fev_{split}"] = list(fev_data[split])
        if gift_data is not None:
            for split in gift_data.keys():
                datasets[f"gift_{split}"] = list(gift_data[split])

    # Run evaluation
    all_results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_type.upper()} PARROT")
        print("=" * 70)

        evaluator = ParrotEvaluator(model_type=model_type)
        results = []

        for dataset_name, dataset in datasets.items():
            print(f"\n  Dataset: {dataset_name}")

            for pred_len in prediction_lengths:
                result = evaluate_on_dataset(
                    evaluator=evaluator,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    prediction_length=pred_len,
                    max_series=max_series
                )
                results.append(result)

                if result.num_series > 0:
                    print(f"    h={pred_len}: MSE={result.metrics['MSE']:.4f}, "
                          f"MAE={result.metrics['MAE']:.4f}, "
                          f"sMAPE={result.metrics['sMAPE']:.2f}")

        all_results[model_type] = results

        # Print summary for this model
        if results:
            avg_metrics = {}
            for r in results:
                if r.num_series > 0:
                    for metric, value in r.metrics.items():
                        if metric not in avg_metrics:
                            avg_metrics[metric] = []
                        avg_metrics[metric].append(value)

            print(f"\n  {model_type.upper()} AVERAGE METRICS:")
            for metric, values in avg_metrics.items():
                print(f"    {metric}: {np.mean(values):.4f}")

    # Save results
    if save_results:
        output_dir = Path("./results")
        output_dir.mkdir(exist_ok=True)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "models": model_types,
            "prediction_lengths": prediction_lengths,
            "results": {}
        }

        for model_type, results in all_results.items():
            summary["results"][model_type] = [
                {
                    "dataset": r.dataset,
                    "prediction_length": r.prediction_length,
                    "num_series": r.num_series,
                    "metrics": r.metrics
                }
                for r in results
            ]

        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {output_dir / 'benchmark_results.json'}")

    return all_results


def print_final_summary(results: Dict[str, List[BenchmarkResult]]):
    """Print final comparison summary."""
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 70)

    comparison = {}

    for model_type, model_results in results.items():
        avg_metrics = {}
        for r in model_results:
            if r.num_series > 0:
                for metric, value in r.metrics.items():
                    if metric not in avg_metrics:
                        avg_metrics[metric] = []
                    avg_metrics[metric].append(value)

        comparison[model_type] = {k: np.mean(v) for k, v in avg_metrics.items()}

    # Print comparison table
    metrics = ["MSE", "MAE", "RMSE", "sMAPE"]
    print(f"\n{'Model':<20} " + " ".join(f"{m:>12}" for m in metrics))
    print("-" * 70)

    for model_type, avg_metrics in comparison.items():
        values = " ".join(f"{avg_metrics.get(m, 0):>12.4f}" for m in metrics)
        print(f"{model_type:<20} {values}")

    # Determine best model
    print("\n" + "-" * 70)
    print("Best Model by Metric:")
    for metric in metrics:
        best_model = min(comparison.keys(),
                         key=lambda m: comparison[m].get(metric, float('inf')))
        print(f"  {metric}: {best_model}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run parrot benchmark evaluation")
    parser.add_argument("--quick", action="store_true", help="Quick test run")
    parser.add_argument("--max-series", type=int, default=50, help="Max series per dataset")

    args = parser.parse_args()

    if args.quick:
        max_series = 10
        pred_lengths = [24]
    else:
        max_series = args.max_series
        pred_lengths = [24, 48]

    results = run_full_evaluation(
        model_types=["context", "derivative"],
        prediction_lengths=pred_lengths,
        max_series=max_series,
        save_results=True
    )

    print_final_summary(results)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nNext steps for submission:")
    print("1. FEV-Bench: Submit to https://huggingface.co/spaces/autogluon/fev-bench")
    print("2. GIFT-Eval: Submit PR to https://github.com/SalesforceAIResearch/gift-eval")
    print("\nNote: Full benchmark requires Python 3.10+ for the fev package.")
