"""
Derivative Parroting Forecast Benchmark

A variant of context parroting that mimics derivative behavior rather than values.
Instead of parroting the continuation values, this algorithm:
1. Finds matching patterns in the DERIVATIVE space
2. Uses the matched derivative pattern to forecast forward recurrently:
   y_{t+1} = y_t + d_mimicked * δt

This approach is inspired by how parrots might learn motion patterns by mimicking
the rate of change rather than absolute positions.
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Optional, Literal


class DerivativeParrotForecaster:
    """
    A pattern matching forecaster that parrots derivative behavior.

    This algorithm:
    1. Computes accurate derivatives from the time series
    2. Finds the best matching derivative pattern in history
    3. Uses that derivative pattern to forecast forward recurrently

    Parameters
    ----------
    min_pattern_length : int, default=30
        Minimum length of derivative pattern to match
    derivative_method : {'savgol', 'central', 'forward'}, default='savgol'
        Method for computing derivatives:
        - 'savgol': Savitzky-Golay filter (smoothest, most accurate)
        - 'central': Central finite difference (balanced)
        - 'forward': Forward finite difference (simple)
    savgol_window : int, default=5
        Window length for Savitzky-Golay filter (must be odd)
    savgol_order : int, default=3
        Polynomial order for Savitzky-Golay filter
    """

    def __init__(
        self,
        min_pattern_length: int = 30,
        derivative_method: Literal['savgol', 'central', 'forward'] = 'savgol',
        savgol_window: int = 5,
        savgol_order: int = 3
    ):
        self.min_pattern_length = min_pattern_length
        self.derivative_method = derivative_method
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order

    def compute_derivative(self, time_series: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Compute accurate derivatives of the time series.

        Parameters
        ----------
        time_series : np.ndarray
            Input time series
        dt : float, default=1.0
            Time step between observations

        Returns
        -------
        np.ndarray
            Derivative at each time point
        """
        if self.derivative_method == 'savgol':
            # Savitzky-Golay filter provides smooth, accurate derivatives
            # This is preferred for noisy data
            derivative = savgol_filter(
                time_series,
                window_length=self.savgol_window,
                polyorder=self.savgol_order,
                deriv=1,
                delta=dt
            )
        elif self.derivative_method == 'central':
            # Central finite difference: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
            derivative = np.zeros_like(time_series)
            derivative[1:-1] = (time_series[2:] - time_series[:-2]) / (2 * dt)
            # Forward/backward difference at boundaries
            derivative[0] = (time_series[1] - time_series[0]) / dt
            derivative[-1] = (time_series[-1] - time_series[-2]) / dt
        elif self.derivative_method == 'forward':
            # Forward finite difference: f'(x) ≈ [f(x+h) - f(x)] / h
            derivative = np.diff(time_series, prepend=time_series[0]) / dt
        else:
            raise ValueError(f"Unknown derivative method: {self.derivative_method}")

        return derivative

    def find_best_derivative_match(
        self,
        derivative_context: np.ndarray,
        derivative_query: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find the position in derivative context that best matches the query pattern.

        Parameters
        ----------
        derivative_context : np.ndarray
            Historical derivative data to search
        derivative_query : np.ndarray
            Recent derivative pattern to match

        Returns
        -------
        best_idx : int
            Starting index of best matching derivative subsequence
        correlation : float
            Correlation coefficient of the best match
        """
        if len(derivative_query) > len(derivative_context):
            raise ValueError("Query length cannot exceed context length")

        # Normalize query for correlation
        query_norm = (derivative_query - np.mean(derivative_query)) / (np.std(derivative_query) + 1e-8)

        best_correlation = -np.inf
        best_idx = 0

        # Slide through context to find best match
        for i in range(len(derivative_context) - len(derivative_query) + 1):
            window = derivative_context[i:i + len(derivative_query)]
            window_norm = (window - np.mean(window)) / (np.std(window) + 1e-8)

            # Compute correlation
            correlation = np.corrcoef(query_norm, window_norm)[0, 1]

            if correlation > best_correlation:
                best_correlation = correlation
                best_idx = i

        return best_idx, best_correlation

    def forecast(
        self,
        time_series: np.ndarray,
        horizon: int,
        dt: float = 1.0,
        pattern_length: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate forecast by parroting derivative patterns.

        The forecast is generated recurrently:
        y_{t+1} = y_t + d_mimicked * δt

        Parameters
        ----------
        time_series : np.ndarray
            Historical time series data (context)
        horizon : int
            Number of steps ahead to forecast
        dt : float, default=1.0
            Time step between observations
        pattern_length : int, optional
            Length of recent derivative pattern to match

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

        # Compute derivatives of the full time series
        derivatives = self.compute_derivative(time_series, dt)

        # Use the most recent derivative pattern as the query
        query_derivatives = derivatives[-pattern_length:]

        # Search in the earlier part of the derivative context
        search_derivatives = derivatives[:-pattern_length]

        # Find best matching derivative subsequence
        match_idx, correlation = self.find_best_derivative_match(
            search_derivatives,
            query_derivatives
        )

        # Extract the derivative pattern that will be mimicked
        continuation_start = match_idx + pattern_length
        continuation_end = min(continuation_start + horizon, len(search_derivatives))
        available_horizon = continuation_end - continuation_start

        # Get the mimicked derivative pattern
        mimicked_derivatives = search_derivatives[continuation_start:continuation_end].copy()

        # If we don't have enough derivative continuation, extend with the last derivative
        if available_horizon < horizon:
            padding = np.full(
                horizon - available_horizon,
                mimicked_derivatives[-1] if len(mimicked_derivatives) > 0 else derivatives[-1]
            )
            mimicked_derivatives = np.concatenate([mimicked_derivatives, padding])

        # Generate forecast recurrently: y_{t+1} = y_t + d_mimicked * δt
        forecast = np.zeros(horizon)
        current_value = time_series[-1]  # Start from the last known value

        for i in range(horizon):
            # Apply the mimicked derivative
            current_value = current_value + mimicked_derivatives[i] * dt
            forecast[i] = current_value

        metadata = {
            'match_idx': match_idx,
            'pattern_length': pattern_length,
            'derivative_correlation': correlation,
            'continuation_available': available_horizon,
            'horizon_requested': horizon,
            'derivative_method': self.derivative_method,
            'dt': dt,
            'mean_mimicked_derivative': np.mean(mimicked_derivatives),
            'std_mimicked_derivative': np.std(mimicked_derivatives)
        }

        return forecast, metadata

    def get_derivative_statistics(self, time_series: np.ndarray, dt: float = 1.0) -> dict:
        """
        Compute statistics about the derivative behavior of the time series.

        Parameters
        ----------
        time_series : np.ndarray
            Input time series
        dt : float, default=1.0
            Time step between observations

        Returns
        -------
        dict
            Statistics including mean, std, min, max of derivatives
        """
        derivatives = self.compute_derivative(time_series, dt)

        return {
            'mean': np.mean(derivatives),
            'std': np.std(derivatives),
            'min': np.min(derivatives),
            'max': np.max(derivatives),
            'median': np.median(derivatives),
            'range': np.max(derivatives) - np.min(derivatives)
        }


def compare_methods(time_series: np.ndarray, horizon: int, dt: float = 1.0) -> dict:
    """
    Compare different derivative computation methods.

    Parameters
    ----------
    time_series : np.ndarray
        Historical time series data
    horizon : int
        Forecast horizon
    dt : float, default=1.0
        Time step

    Returns
    -------
    dict
        Forecasts and metadata for each method
    """
    methods = ['savgol', 'central', 'forward']
    results = {}

    for method in methods:
        forecaster = DerivativeParrotForecaster(
            min_pattern_length=30,
            derivative_method=method
        )
        forecast, metadata = forecaster.forecast(time_series, horizon, dt)
        results[method] = {
            'forecast': forecast,
            'metadata': metadata
        }

    return results


if __name__ == "__main__":
    print("Derivative Parroting Forecast Benchmark")
    print("=" * 50)

    # Generate synthetic data (chaotic-like behavior)
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    dt = t[1] - t[0]

    # Nonlinear oscillation with chaotic-like features
    signal = np.sin(t) + 0.5 * np.sin(3 * t) + 0.2 * np.cos(5 * t)
    signal += 0.05 * np.random.randn(len(t))  # Add small noise

    # Split into context and test
    context_data = signal[:700]
    test_data = signal[700:800]

    print(f"\nTime series properties:")
    print(f"  Length: {len(context_data)}")
    print(f"  dt: {dt:.4f}")

    # Test different derivative methods
    print("\n" + "=" * 50)
    print("Comparing Derivative Methods:")
    print("=" * 50)

    for method in ['savgol', 'central', 'forward']:
        print(f"\nMethod: {method.upper()}")
        print("-" * 40)

        forecaster = DerivativeParrotForecaster(
            min_pattern_length=30,
            derivative_method=method
        )

        # Get derivative statistics
        deriv_stats = forecaster.get_derivative_statistics(context_data, dt)
        print(f"  Derivative Statistics:")
        print(f"    Mean: {deriv_stats['mean']:.4f}")
        print(f"    Std:  {deriv_stats['std']:.4f}")
        print(f"    Range: [{deriv_stats['min']:.4f}, {deriv_stats['max']:.4f}]")

        # Generate forecast
        forecast, metadata = forecaster.forecast(
            time_series=context_data,
            horizon=len(test_data),
            dt=dt,
            pattern_length=50
        )

        # Evaluate
        from context_parrot_benchmark import evaluate_forecast
        metrics = evaluate_forecast(test_data, forecast)

        print(f"\n  Forecast Metadata:")
        print(f"    Match at index: {metadata['match_idx']}")
        print(f"    Derivative correlation: {metadata['derivative_correlation']:.4f}")
        print(f"    Mean mimicked derivative: {metadata['mean_mimicked_derivative']:.4f}")

        print(f"\n  Performance:")
        print(f"    sMAPE: {metrics['sMAPE']:.2f}")
        print(f"    MAE: {metrics['MAE']:.4f}")
        print(f"    RMSE: {metrics['RMSE']:.4f}")

    print("\n" + "=" * 50)
    print("Derivative parroting complete!")
    print("\nThis approach mimics the RATE OF CHANGE rather than absolute values.")
    print("Forecast evolves as: y_{t+1} = y_t + d_mimicked * δt")
    print("Where d_mimicked comes from matching historical derivative patterns.")
