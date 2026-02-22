"""
FEV-Bench Evaluation for Parrot Forecasters

This script evaluates the parrot forecasting models against the FEV-Bench benchmark,
which includes 100 forecasting tasks across 7 domains.

Reference: https://github.com/autogluon/fev
Paper: "fev-bench: A Realistic Benchmark for Time Series Forecasting"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import json
from pathlib import Path

# Import parrot forecasters
from context_parrot_benchmark import ContextParrotForecaster, evaluate_forecast
from derivative_parrot_benchmark import DerivativeParrotForecaster
from multivariate_parrot import (
    MultivarateContextParrotForecaster,
    MultivariateDerivativeParrotForecaster
)

try:
    import fev
    from fev import Dataset, Task
    FEV_AVAILABLE = True
except ImportError:
    FEV_AVAILABLE = False
    print("Warning: fev package not installed. Run: pip install fev")


@dataclass
class ParrotForecastResult:
    """Container for forecast results."""
    task_name: str
    predictions: np.ndarray
    metadata: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None


class ParrotFEVAdapter:
    """
    Adapter to run Parrot forecasters on FEV-Bench tasks.

    This adapter wraps the parrot forecasting models to work with the
    FEV benchmark format and evaluation pipeline.
    """

    def __init__(
        self,
        model_type: str = "context",  # "context", "derivative", "multivariate_context", "multivariate_derivative"
        min_pattern_length: int = 30,
        derivative_method: str = "savgol",
    ):
        self.model_type = model_type
        self.min_pattern_length = min_pattern_length
        self.derivative_method = derivative_method

        # Initialize appropriate forecaster
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

    def predict(
        self,
        context: np.ndarray,
        prediction_length: int,
        **kwargs
    ) -> np.ndarray:
        """
        Generate predictions in FEV-compatible format.

        Parameters
        ----------
        context : np.ndarray
            Historical time series data (univariate or multivariate)
        prediction_length : int
            Number of steps to forecast

        Returns
        -------
        np.ndarray
            Point predictions of shape (prediction_length,) or (prediction_length, n_targets)
        """
        # Handle univariate vs multivariate
        if context.ndim == 1:
            # Univariate case
            if "multivariate" in self.model_type:
                # Reshape to (n, 1) for multivariate forecaster
                context = context.reshape(-1, 1)

        # Check minimum length requirements
        min_required = self.min_pattern_length + prediction_length
        if len(context) < min_required:
            # Fall back to naive forecast (last value)
            if context.ndim == 1:
                return np.full(prediction_length, context[-1])
            else:
                return np.full((prediction_length, context.shape[1]), context[-1, :])

        try:
            # Generate forecast
            if "multivariate" in self.model_type:
                if context.ndim == 1:
                    context = context.reshape(-1, 1)
                forecast, _ = self.forecaster.forecast(
                    time_series=context,
                    horizon=prediction_length,
                    pattern_length=min(self.min_pattern_length, len(context) // 3)
                )
            else:
                if context.ndim > 1:
                    # Use first column for univariate forecaster
                    context = context[:, 0]
                forecast, _ = self.forecaster.forecast(
                    time_series=context,
                    horizon=prediction_length,
                    pattern_length=min(self.min_pattern_length, len(context) // 3)
                )
            return forecast

        except Exception as e:
            # Fallback to naive forecast
            print(f"Warning: Forecast failed ({e}), using naive fallback")
            if context.ndim == 1:
                return np.full(prediction_length, context[-1])
            else:
                return np.full(prediction_length, context[-1, 0])

    def predict_quantiles(
        self,
        context: np.ndarray,
        prediction_length: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        **kwargs
    ) -> np.ndarray:
        """
        Generate probabilistic predictions (quantile forecasts).

        Since parrot is a deterministic method, we estimate uncertainty
        based on pattern match quality and historical variance.

        Parameters
        ----------
        context : np.ndarray
            Historical time series data
        prediction_length : int
            Number of steps to forecast
        quantiles : List[float]
            Quantile levels to predict

        Returns
        -------
        np.ndarray
            Quantile predictions of shape (prediction_length, len(quantiles))
        """
        # Get point forecast
        point_forecast = self.predict(context, prediction_length, **kwargs)

        # Estimate uncertainty from historical variance
        if context.ndim == 1:
            historical_std = np.std(context)
        else:
            historical_std = np.std(context[:, 0])

        # Scale uncertainty by horizon (increases with prediction distance)
        horizon_scale = np.sqrt(1 + np.arange(prediction_length) * 0.1)
        uncertainty = historical_std * horizon_scale * 0.5

        # Generate quantile forecasts
        from scipy.stats import norm
        quantile_forecasts = np.zeros((prediction_length, len(quantiles)))

        for i, q in enumerate(quantiles):
            z_score = norm.ppf(q)
            quantile_forecasts[:, i] = point_forecast + z_score * uncertainty

        return quantile_forecasts


def run_fev_benchmark(
    model_types: List[str] = ["context", "derivative"],
    num_tasks: Optional[int] = None,
    save_results: bool = True,
    output_dir: str = "./fev_results"
) -> Dict[str, Dict]:
    """
    Run FEV-Bench evaluation on parrot forecasters.

    Parameters
    ----------
    model_types : List[str]
        Which parrot models to evaluate
    num_tasks : int, optional
        Number of tasks to evaluate (None = all 100)
    save_results : bool
        Whether to save results to disk
    output_dir : str
        Directory to save results

    Returns
    -------
    Dict[str, Dict]
        Results for each model type
    """
    if not FEV_AVAILABLE:
        print("FEV package not available. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "fev"])
        import fev

    print("=" * 70)
    print("FEV-BENCH EVALUATION")
    print("=" * 70)

    # Load FEV benchmark tasks
    print("\nLoading FEV-Bench tasks from Hugging Face...")
    try:
        tasks = fev.get_benchmark_tasks()
        print(f"Loaded {len(tasks)} tasks")
    except Exception as e:
        print(f"Error loading tasks: {e}")
        print("Falling back to manual dataset loading...")
        tasks = load_fev_tasks_manual()

    if num_tasks is not None:
        tasks = tasks[:num_tasks]
        print(f"Evaluating on {num_tasks} tasks (subset)")

    # Results storage
    all_results = {}

    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_type.upper()} PARROT")
        print("=" * 70)

        adapter = ParrotFEVAdapter(model_type=model_type)
        model_results = []

        for task in tqdm(tasks, desc=f"Running {model_type}"):
            try:
                result = evaluate_single_task(adapter, task)
                model_results.append(result)
            except Exception as e:
                print(f"\nError on task {task.name}: {e}")
                continue

        # Aggregate metrics
        if model_results:
            metrics_df = pd.DataFrame([r.metrics for r in model_results if r.metrics])
            avg_metrics = metrics_df.mean().to_dict()

            print(f"\n{model_type.upper()} Average Metrics:")
            for metric, value in avg_metrics.items():
                print(f"  {metric}: {value:.4f}")

            all_results[model_type] = {
                "individual_results": model_results,
                "average_metrics": avg_metrics,
                "num_tasks_evaluated": len(model_results)
            }

    # Save results
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        summary = {
            model: {
                "average_metrics": results["average_metrics"],
                "num_tasks": results["num_tasks_evaluated"]
            }
            for model, results in all_results.items()
        }

        with open(output_path / "fev_results_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return all_results


def evaluate_single_task(adapter: ParrotFEVAdapter, task) -> ParrotForecastResult:
    """Evaluate adapter on a single FEV task."""

    # Get task data
    context = task.get_context()
    target = task.get_target()
    prediction_length = task.prediction_length

    # Generate predictions
    predictions = adapter.predict(context, prediction_length)

    # Compute metrics
    metrics = evaluate_forecast(target, predictions)

    return ParrotForecastResult(
        task_name=task.name,
        predictions=predictions,
        metadata={"prediction_length": prediction_length},
        metrics=metrics
    )


def load_fev_tasks_manual():
    """Manual loading of FEV tasks from Hugging Face datasets."""
    from datasets import load_dataset

    print("Loading FEV datasets from Hugging Face...")
    dataset = load_dataset("autogluon/fev_datasets", split="test")

    # Convert to task objects
    tasks = []
    for item in dataset:
        tasks.append(FEVTaskWrapper(item))

    return tasks


class FEVTaskWrapper:
    """Wrapper to make HuggingFace dataset items look like FEV tasks."""

    def __init__(self, data_item):
        self.data = data_item
        self.name = data_item.get("name", "unknown")
        self.prediction_length = data_item.get("prediction_length", 24)

    def get_context(self):
        return np.array(self.data.get("context", self.data.get("past_values", [])))

    def get_target(self):
        return np.array(self.data.get("target", self.data.get("future_values", [])))


if __name__ == "__main__":
    print("FEV-Bench Evaluation for Parrot Forecasters")
    print("=" * 70)

    # Check if fev is available
    if not FEV_AVAILABLE:
        print("\nThe 'fev' package is required. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "fev"])
        print("Please restart the script after installation.")
    else:
        # Run benchmark
        results = run_fev_benchmark(
            model_types=["context", "derivative"],
            num_tasks=10,  # Start with subset for testing
            save_results=True
        )

        print("\n" + "=" * 70)
        print("FEV-BENCH EVALUATION COMPLETE")
        print("=" * 70)
