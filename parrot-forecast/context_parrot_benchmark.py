"""
Context Parroting Forecast Benchmark

Implementation of the simple pattern matching algorithm described in:
"Zero-shot Forecasting of Chaotic Systems" by Zhang & Gilpin (ICLR 2025)

The algorithm finds the best matching subsequence in the context window
and uses its continuation to forecast future values - essentially "parroting"
patterns from the past.
"""

import numpy as np
from scipy.signal import correlate
from typing import Tuple, Optional


class ContextParrotForecaster:
    """
    A simple pattern matching forecaster that uses context parroting.

    This algorithm:
    1. Takes a context window of historical time series data
    2. Finds the subsequence that best matches the most recent pattern
    3. Uses that matching subsequence's continuation to predict the future

    Parameters
    ----------
    min_pattern_length : int, default=30
        Minimum length of pattern to match (paper uses 30 timepoints = 1 Lyapunov time)
    context_length : int, default=512
        Length of context window to search for patterns
    """

    def __init__(self, min_pattern_length: int = 30, context_length: int = 512):
        self.min_pattern_length = min_pattern_length
        self.context_length = context_length

    def find_best_match(self, context: np.ndarray, query: np.ndarray) -> int:
        """
        Find the position in context that best matches the query pattern.

        Parameters
        ----------
        context : np.ndarray
            Historical time series data to search
        query : np.ndarray
            Recent pattern to match

        Returns
        -------
        int
            Starting index of best matching subsequence in context
        """
        if len(query) > len(context):
            raise ValueError("Query length cannot exceed context length")

        # Normalize query for correlation
        query_norm = (query - np.mean(query)) / (np.std(query) + 1e-8)

        best_correlation = -np.inf
        best_idx = 0

        # Slide through context to find best match
        for i in range(len(context) - len(query) + 1):
            window = context[i:i + len(query)]
            window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)

            # Compute correlation
            correlation = np.corrcoef(query_norm, window_norm)[0, 1]

            if correlation > best_correlation:
                best_correlation = correlation
                best_idx = i

        return best_idx

    def forecast(self,
                 time_series: np.ndarray,
                 horizon: int,
                 pattern_length: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Generate forecast by parroting the best matching context pattern.

        Parameters
        ----------
        time_series : np.ndarray
            Historical time series data (context)
        horizon : int
            Number of steps ahead to forecast
        pattern_length : int, optional
            Length of recent pattern to match. If None, uses min_pattern_length

        Returns
        -------
        forecast : np.ndarray
            Predicted values for the forecast horizon
        metadata : dict
            Information about the forecast (match location, correlation, etc.)
        """
        if pattern_length is None:
            pattern_length = self.min_pattern_length

        if len(time_series) < pattern_length + horizon:
            raise ValueError(
                f"Time series too short. Need at least {pattern_length + horizon} points, "
                f"got {len(time_series)}"
            )

        # Use the most recent data as the pattern to match
        query_pattern = time_series[-pattern_length:]

        # Search in the earlier part of the context (excluding the query itself)
        search_context = time_series[:-pattern_length]

        # Find best matching subsequence
        match_idx = self.find_best_match(search_context, query_pattern)

        # Check if we have enough data after the match for forecasting
        continuation_start = match_idx + pattern_length
        continuation_end = min(continuation_start + horizon, len(search_context))
        available_horizon = continuation_end - continuation_start

        # Extract the continuation (what came after the match)
        forecast = search_context[continuation_start:continuation_end].copy()

        # If we don't have enough continuation, extend with the last value
        if available_horizon < horizon:
            padding = np.full(horizon - available_horizon, forecast[-1] if len(forecast) > 0 else time_series[-1])
            forecast = np.concatenate([forecast, padding])

        # Calculate match quality
        match_window = search_context[match_idx:match_idx + pattern_length]
        query_norm = (query_pattern - np.mean(query_pattern)) / (np.std(query_pattern) + 1e-8)
        match_norm = (match_window - np.mean(match_window)) / (np.std(match_window) + 1e-8)
        correlation = np.corrcoef(query_norm, match_norm)[0, 1]

        metadata = {
            'match_idx': match_idx,
            'pattern_length': pattern_length,
            'correlation': correlation,
            'continuation_available': available_horizon,
            'horizon_requested': horizon
        }

        return forecast, metadata


def evaluate_forecast(true_values: np.ndarray, predicted_values: np.ndarray) -> dict:
    """
    Evaluate forecast performance using metrics from the paper.

    Parameters
    ----------
    true_values : np.ndarray
        Ground truth values
    predicted_values : np.ndarray
        Forecasted values

    Returns
    -------
    dict
        Dictionary containing sMAPE and other error metrics
    """
    # Symmetric Mean Absolute Percentage Error (sMAPE)
    epsilon = 1e-8
    smape = 200 * np.mean(
        np.abs(predicted_values - true_values) /
        (np.abs(true_values) + np.abs(predicted_values) + epsilon)
    )

    # Mean Absolute Error
    mae = np.mean(np.abs(predicted_values - true_values))

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((predicted_values - true_values) ** 2))

    return {
        'sMAPE': smape,
        'MAE': mae,
        'RMSE': rmse
    }


if __name__ == "__main__":
    # Example: Forecast a simple chaotic system (Lorenz attractor simulation)
    print("Context Parroting Forecast Benchmark")
    print("=" * 50)

    # Generate synthetic data (simulating a chaotic time series)
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    # Simple nonlinear oscillation as example
    signal = np.sin(t) + 0.5 * np.sin(3 * t) + 0.1 * np.random.randn(len(t))

    # Split into context and test
    context_data = signal[:700]
    test_data = signal[700:800]

    # Initialize forecaster
    forecaster = ContextParrotForecaster(min_pattern_length=30, context_length=512)

    # Generate forecast
    forecast, metadata = forecaster.forecast(
        time_series=context_data,
        horizon=len(test_data),
        pattern_length=50
    )

    # Evaluate
    metrics = evaluate_forecast(test_data, forecast)

    print(f"\nForecast Metadata:")
    print(f"  Match found at index: {metadata['match_idx']}")
    print(f"  Pattern correlation: {metadata['correlation']:.4f}")
    print(f"  Pattern length: {metadata['pattern_length']}")

    print(f"\nForecast Performance:")
    print(f"  sMAPE: {metrics['sMAPE']:.2f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")

    print("\nBenchmark implementation complete!")
    print("This simple pattern matching approach forms the basis of 'context parroting'")
    print("observed in foundation models like Chronos when forecasting chaotic systems.")
