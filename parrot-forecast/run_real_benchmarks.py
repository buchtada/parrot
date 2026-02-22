#!/usr/bin/env python3
"""
Run Real FEV-Bench and GIFT-Eval Benchmarks

This script runs the parrot forecasters against the actual benchmark datasets
using the fev library (requires Python 3.10+).

Usage:
    source venv310/bin/activate
    python run_real_benchmarks.py --quick      # Quick test (5 tasks)
    python run_real_benchmarks.py              # Full benchmark
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import fev
import fev
from fev import Task

# Import parrot forecasters
from context_parrot_benchmark import ContextParrotForecaster, evaluate_forecast
from derivative_parrot_benchmark import DerivativeParrotForecaster


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    task_name: str
    model: str
    prediction_length: int
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


class ParrotPredictor:
    """Wrapper for parrot forecasters compatible with fev."""

    def __init__(self, model_type: str = "context", min_pattern_length: int = 30):
        self.model_type = model_type
        self.min_pattern_length = min_pattern_length
        self.name = f"parrot_{model_type}"

        if model_type == "context":
            self.forecaster = ContextParrotForecaster(min_pattern_length=min_pattern_length)
        elif model_type == "derivative":
            self.forecaster = DerivativeParrotForecaster(
                min_pattern_length=min_pattern_length,
                derivative_method='savgol'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def predict(
        self,
        context: np.ndarray,
        prediction_length: int,
        **kwargs
    ) -> np.ndarray:
        """Generate point forecast."""
        # Handle different input shapes
        if context.ndim > 1:
            # Use first column for univariate
            context = context[:, 0] if context.shape[1] > 0 else context.flatten()

        # Check minimum length
        min_required = self.min_pattern_length + prediction_length
        if len(context) < min_required:
            return np.full(prediction_length, context[-1] if len(context) > 0 else 0)

        try:
            pattern_length = min(self.min_pattern_length, max(10, len(context) // 4))
            forecast, _ = self.forecaster.forecast(
                time_series=context,
                horizon=prediction_length,
                pattern_length=pattern_length
            )
            return forecast
        except Exception as e:
            # Fallback to naive
            return np.full(prediction_length, context[-1] if len(context) > 0 else 0)

    def predict_quantiles(
        self,
        context: np.ndarray,
        prediction_length: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        **kwargs
    ) -> np.ndarray:
        """Generate quantile forecasts."""
        point_forecast = self.predict(context, prediction_length, **kwargs)

        # Estimate uncertainty from context variance
        if context.ndim > 1:
            context = context[:, 0] if context.shape[1] > 0 else context.flatten()

        historical_std = np.std(context) if len(context) > 1 else 1.0

        # Scale uncertainty by horizon
        horizon_scale = np.sqrt(1 + np.arange(prediction_length) * 0.1)
        uncertainty = historical_std * horizon_scale * 0.3

        from scipy.stats import norm
        quantile_forecasts = np.zeros((prediction_length, len(quantiles)))

        for i, q in enumerate(quantiles):
            z_score = norm.ppf(q)
            quantile_forecasts[:, i] = point_forecast + z_score * uncertainty

        return quantile_forecasts


def run_fev_benchmark(
    model_types: List[str] = ["context", "derivative"],
    num_tasks: Optional[int] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run FEV-Bench evaluation.

    Parameters
    ----------
    model_types : List[str]
        Which parrot variants to evaluate
    num_tasks : int, optional
        Number of tasks (None = all 100)
    save_results : bool
        Whether to save results

    Returns
    -------
    Dict containing results for each model
    """
    print("=" * 70)
    print("FEV-BENCH EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load FEV benchmark tasks
    print("\nLoading FEV benchmark tasks...")
    try:
        tasks = fev.get_benchmark_tasks()
        print(f"Loaded {len(tasks)} tasks")

        if num_tasks is not None:
            tasks = tasks[:num_tasks]
            print(f"Using subset: {num_tasks} tasks")
    except Exception as e:
        print(f"Error loading tasks: {e}")
        print("Falling back to dataset loading...")
        tasks = load_fev_tasks_from_dataset()

    all_results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Model: {model_type.upper()} PARROT")
        print("=" * 70)

        predictor = ParrotPredictor(model_type=model_type)
        results = []

        for task in tqdm(tasks, desc=f"Evaluating {model_type}"):
            try:
                result = evaluate_task(predictor, task)
                results.append(result)
            except Exception as e:
                print(f"\n  Error on task {getattr(task, 'name', 'unknown')}: {e}")
                continue

        if results:
            # Filter out error results and compute aggregate metrics
            valid_results = [r for r in results if 'MSE' in r.metrics]
            if valid_results:
                metrics_df = pd.DataFrame([r.metrics for r in valid_results])
                avg_metrics = metrics_df.mean(numeric_only=True).to_dict()
            else:
                avg_metrics = {"note": "no valid results"}

            print(f"\n{model_type.upper()} Average Metrics:")
            for metric, value in avg_metrics.items():
                print(f"  {metric}: {value:.4f}")

            all_results[model_type] = {
                "results": [
                    {
                        "task": r.task_name,
                        "prediction_length": r.prediction_length,
                        "metrics": r.metrics
                    }
                    for r in valid_results
                ],
                "average_metrics": avg_metrics,
                "num_tasks": len(valid_results),
                "total_tasks": len(results)
            }

    # Save results
    if save_results:
        output_dir = Path("./results/fev")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "fev_benchmark_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "num_tasks": len(tasks),
                "models": all_results
            }, f, indent=2)

        print(f"\nResults saved to {output_dir}")

    return all_results


def evaluate_task(predictor: ParrotPredictor, task) -> BenchmarkResult:
    """Evaluate predictor on a single FEV task."""

    # Get task data - handle different API versions
    if hasattr(task, 'past_target'):
        context = np.array(task.past_target)
        target = np.array(task.future_target) if hasattr(task, 'future_target') else None
        prediction_length = task.prediction_length if hasattr(task, 'prediction_length') else len(target) if target is not None else 24
    elif hasattr(task, 'get_context'):
        context = np.array(task.get_context())
        target = np.array(task.get_target()) if hasattr(task, 'get_target') else None
        prediction_length = getattr(task, 'prediction_length', 24)
    else:
        # Try dictionary-like access for HuggingFace dataset items
        # FEV datasets use 'target' for the full time series
        full_target = task.get('target', task.get('past_target', []))
        if isinstance(full_target, (list, np.ndarray)) and len(full_target) > 0:
            full_target = np.array(full_target)
            # Use last 20% as target, rest as context
            prediction_length = task.get('prediction_length', max(24, len(full_target) // 5))
            if len(full_target) > prediction_length:
                context = full_target[:-prediction_length]
                target = full_target[-prediction_length:]
            else:
                context = full_target
                target = None
                prediction_length = 24
        else:
            context = np.array(task.get('past_target', task.get('context', [])))
            target = np.array(task.get('future_target', []))
            prediction_length = task.get('prediction_length', len(target) if len(target) > 0 else 24)

    # Ensure context is valid
    if len(context) == 0:
        return BenchmarkResult(
            task_name="invalid",
            model=predictor.name,
            prediction_length=prediction_length,
            metrics={"error": "empty_context"},
            metadata={}
        )

    # Generate predictions
    predictions = predictor.predict(context, prediction_length)

    # Compute metrics
    if target is not None and len(target) > 0:
        metrics = compute_metrics(target, predictions)
    else:
        metrics = {"note": "no_target_available"}

    task_name = task.get('_config', task.get('name', 'unknown')) if isinstance(task, dict) else getattr(task, 'name', 'unknown')

    return BenchmarkResult(
        task_name=task_name,
        model=predictor.name,
        prediction_length=prediction_length,
        metrics=metrics,
        metadata={}
    )


def compute_metrics(target: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    # Ensure same length
    min_len = min(len(target), len(predictions))
    target = target[:min_len]
    predictions = predictions[:min_len]

    epsilon = 1e-8

    mse = float(np.mean((predictions - target) ** 2))
    mae = float(np.mean(np.abs(predictions - target)))
    rmse = float(np.sqrt(mse))

    smape = float(200 * np.mean(
        np.abs(predictions - target) /
        (np.abs(target) + np.abs(predictions) + epsilon)
    ))

    # MASE (using naive seasonal baseline)
    naive_error = np.mean(np.abs(np.diff(target))) if len(target) > 1 else 1.0
    mase = mae / (naive_error + epsilon)

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "MASE": float(mase)
    }


def load_fev_tasks_from_dataset(configs: Optional[List[str]] = None):
    """Load FEV tasks directly from HuggingFace dataset."""
    from datasets import load_dataset

    # Default configs to test (subset of all 100)
    if configs is None:
        configs = [
            'ETT_1H',           # Energy
            'solar_1D',         # Solar power
            'hospital',         # Hospital admissions
            'restaurant',       # Restaurant demand
            'walmart',          # Retail
            'jena_weather_1H',  # Weather
            'australian_tourism',  # Tourism
            'fred_md_2025',     # Economic indicators
        ]

    print(f"Loading FEV datasets from HuggingFace ({len(configs)} configs)...")

    tasks = []
    for config_name in tqdm(configs, desc="Loading datasets"):
        try:
            dataset = load_dataset("autogluon/fev_datasets", config_name, trust_remote_code=True)
            for split_name in dataset.keys():
                for item in dataset[split_name]:
                    item['_config'] = config_name
                    item['_split'] = split_name
                    tasks.append(item)
        except Exception as e:
            print(f"  Warning: Could not load {config_name}: {e}")

    print(f"Loaded {len(tasks)} tasks from {len(configs)} datasets")
    return tasks


def run_gift_eval(
    model_types: List[str] = ["context", "derivative"],
    max_series: int = 100,
    prediction_lengths: List[int] = [24, 48, 96],
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run GIFT-Eval benchmark.

    Parameters
    ----------
    model_types : List[str]
        Which parrot variants to evaluate
    max_series : int
        Max series per dataset
    prediction_lengths : List[int]
        Forecast horizons to evaluate
    save_results : bool
        Whether to save results

    Returns
    -------
    Dict containing results
    """
    from datasets import load_dataset

    print("\n" + "=" * 70)
    print("GIFT-EVAL BENCHMARK")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load GIFT-Eval dataset
    print("\nLoading GIFT-Eval datasets...")
    try:
        dataset = load_dataset("Salesforce/GiftEval", trust_remote_code=True)
        print(f"Loaded splits: {list(dataset.keys())}")
    except Exception as e:
        print(f"Error loading GIFT-Eval: {e}")
        return {}

    all_results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Model: {model_type.upper()} PARROT")
        print("=" * 70)

        predictor = ParrotPredictor(model_type=model_type)
        model_results = []

        for split_name in dataset.keys():
            print(f"\n  Dataset: {split_name}")
            split_data = list(dataset[split_name])[:max_series]

            for pred_len in prediction_lengths:
                predictions_list = []
                targets_list = []

                for item in tqdm(split_data, desc=f"    h={pred_len}", leave=False):
                    # Extract time series
                    values = np.array(item.get("target", item.get("values", [])))

                    if len(values) < pred_len + predictor.min_pattern_length:
                        continue

                    context = values[:-pred_len]
                    target = values[-pred_len:]

                    pred = predictor.predict(context, pred_len)
                    predictions_list.append(pred)
                    targets_list.append(target)

                if predictions_list:
                    # Compute metrics
                    all_preds = np.concatenate(predictions_list)
                    all_targets = np.concatenate(targets_list)
                    metrics = compute_metrics(all_targets, all_preds)

                    model_results.append({
                        "dataset": split_name,
                        "prediction_length": pred_len,
                        "num_series": len(predictions_list),
                        "metrics": metrics
                    })

                    print(f"      h={pred_len}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}")

        if model_results:
            # Compute averages
            avg_metrics = {}
            for r in model_results:
                for k, v in r["metrics"].items():
                    if k not in avg_metrics:
                        avg_metrics[k] = []
                    avg_metrics[k].append(v)
            avg_metrics = {k: np.mean(v) for k, v in avg_metrics.items()}

            print(f"\n{model_type.upper()} Average Metrics:")
            for metric, value in avg_metrics.items():
                print(f"  {metric}: {value:.4f}")

            all_results[model_type] = {
                "results": model_results,
                "average_metrics": avg_metrics
            }

    # Save results
    if save_results:
        output_dir = Path("./results/gift")
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "gift_eval_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "models": all_results
            }, f, indent=2)

        # Generate config.json for submission
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
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nResults saved to {output_dir}")

    return all_results


def print_comparison_summary(fev_results: Dict, gift_results: Dict):
    """Print final comparison summary."""
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK COMPARISON")
    print("=" * 70)

    metrics = ["MSE", "MAE", "RMSE", "sMAPE"]

    print("\nFEV-BENCH Results:")
    print(f"{'Model':<20} " + " ".join(f"{m:>10}" for m in metrics))
    print("-" * 60)
    for model, data in fev_results.items():
        avg = data.get("average_metrics", {})
        values = " ".join(f"{avg.get(m, 0):>10.4f}" for m in metrics)
        print(f"{model:<20} {values}")

    if gift_results:
        print("\nGIFT-EVAL Results:")
        print(f"{'Model':<20} " + " ".join(f"{m:>10}" for m in metrics))
        print("-" * 60)
        for model, data in gift_results.items():
            avg = data.get("average_metrics", {})
            values = " ".join(f"{avg.get(m, 0):>10.4f}" for m in metrics)
            print(f"{model:<20} {values}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FEV-Bench and GIFT-Eval benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 FEV tasks)")
    parser.add_argument("--fev-only", action="store_true", help="Run FEV-Bench only")
    parser.add_argument("--gift-only", action="store_true", help="Run GIFT-Eval only")
    parser.add_argument("--max-series", type=int, default=100, help="Max series for GIFT-Eval")

    args = parser.parse_args()

    num_tasks = 5 if args.quick else None
    max_series = 20 if args.quick else args.max_series

    fev_results = {}
    gift_results = {}

    if not args.gift_only:
        fev_results = run_fev_benchmark(
            model_types=["context", "derivative"],
            num_tasks=num_tasks,
            save_results=True
        )

    if not args.fev_only:
        gift_results = run_gift_eval(
            model_types=["context", "derivative"],
            max_series=max_series,
            prediction_lengths=[24, 48] if args.quick else [24, 48, 96],
            save_results=True
        )

    print_comparison_summary(fev_results, gift_results)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nSubmission links:")
    print("  FEV-Bench: https://huggingface.co/spaces/autogluon/fev-bench")
    print("  GIFT-Eval: https://github.com/SalesforceAIResearch/gift-eval")
