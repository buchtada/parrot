"""
Multivariate Parroting Forecasters

Extension of parroting algorithms to handle multivariate inputs:
- Multiple input signals (e.g., x, y, z coordinates)
- Pattern matching across all dimensions simultaneously
- Forecast for a single target dimension

This is particularly useful for systems like:
- Lorenz attractor (x, y, z) -> predict x using patterns from all three
- Weather data (temp, pressure, humidity) -> predict temp using all signals
- Financial markets (price, volume, volatility) -> predict price using all
"""

import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Optional, Literal, Union


class MultivarateContextParrotForecaster:
    """
    Pattern matching across multiple input signals.

    Finds patterns that match across ALL dimensions simultaneously,
    then uses the continuation to forecast the target dimension.

    Parameters
    ----------
    min_pattern_length : int, default=30
        Minimum length of pattern to match
    target_dim : int, default=0
        Which dimension to forecast (0-indexed)
    """

    def __init__(self, min_pattern_length: int = 30, target_dim: int = 0):
        self.min_pattern_length = min_pattern_length
        self.target_dim = target_dim

    def find_best_multivariate_match(
        self,
        context: np.ndarray,
        query: np.ndarray
    ) -> Tuple[int, float]:
        """
        Find best matching pattern across all dimensions.

        Parameters
        ----------
        context : np.ndarray, shape (n_timesteps, n_dims)
            Historical multivariate data
        query : np.ndarray, shape (pattern_length, n_dims)
            Recent multivariate pattern to match

        Returns
        -------
        best_idx : int
            Starting index of best match
        correlation : float
            Average correlation across all dimensions
        """
        pattern_length, n_dims = query.shape
        n_context = context.shape[0]

        if pattern_length > n_context:
            raise ValueError("Query length exceeds context length")

        best_correlation = -np.inf
        best_idx = 0

        # Normalize query patterns for each dimension
        query_norm = np.zeros_like(query)
        for d in range(n_dims):
            query_norm[:, d] = (query[:, d] - np.mean(query[:, d])) / (np.std(query[:, d]) + 1e-8)

        # Slide through context
        for i in range(n_context - pattern_length + 1):
            window = context[i:i + pattern_length, :]

            # Compute correlation for each dimension
            correlations = []
            for d in range(n_dims):
                window_norm = (window[:, d] - np.mean(window[:, d])) / (np.std(window[:, d]) + 1e-8)
                corr = np.corrcoef(query_norm[:, d], window_norm)[0, 1]
                correlations.append(corr)

            # Average correlation across all dimensions
            avg_correlation = np.mean(correlations)

            if avg_correlation > best_correlation:
                best_correlation = avg_correlation
                best_idx = i

        return best_idx, best_correlation

    def forecast(
        self,
        time_series: np.ndarray,
        horizon: int,
        pattern_length: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate forecast using multivariate pattern matching.

        Parameters
        ----------
        time_series : np.ndarray, shape (n_timesteps, n_dims)
            Multivariate time series data
        horizon : int
            Forecast horizon
        pattern_length : int, optional
            Length of pattern to match

        Returns
        -------
        forecast : np.ndarray, shape (horizon,)
            Forecast for target dimension only
        metadata : dict
            Information about the match
        """
        if pattern_length is None:
            pattern_length = self.min_pattern_length

        n_timesteps, n_dims = time_series.shape

        if self.target_dim >= n_dims:
            raise ValueError(f"target_dim {self.target_dim} >= n_dims {n_dims}")

        if n_timesteps < pattern_length + horizon:
            raise ValueError(
                f"Time series too short. Need {pattern_length + horizon}, got {n_timesteps}"
            )

        # Extract recent multivariate pattern
        query_pattern = time_series[-pattern_length:, :]

        # Search in earlier context
        search_context = time_series[:-pattern_length, :]

        # Find best match across all dimensions
        match_idx, correlation = self.find_best_multivariate_match(
            search_context,
            query_pattern
        )

        # Extract continuation for TARGET dimension only
        continuation_start = match_idx + pattern_length
        continuation_end = min(continuation_start + horizon, len(search_context))
        available_horizon = continuation_end - continuation_start

        forecast = search_context[continuation_start:continuation_end, self.target_dim].copy()

        # Pad if needed
        if available_horizon < horizon:
            padding = np.full(
                horizon - available_horizon,
                forecast[-1] if len(forecast) > 0 else time_series[-1, self.target_dim]
            )
            forecast = np.concatenate([forecast, padding])

        metadata = {
            'match_idx': match_idx,
            'pattern_length': pattern_length,
            'multivariate_correlation': correlation,
            'n_dimensions': n_dims,
            'target_dimension': self.target_dim,
            'continuation_available': available_horizon
        }

        return forecast, metadata


class MultivariateDerivativeParrotForecaster:
    """
    Multivariate derivative parroting.

    Computes derivatives for each dimension, finds matching derivative patterns
    across all dimensions, then forecasts the target dimension using:
    y_{t+1} = y_t + d_mimicked * δt

    Parameters
    ----------
    min_pattern_length : int, default=30
        Minimum pattern length
    target_dim : int, default=0
        Which dimension to forecast
    derivative_method : {'savgol', 'central', 'forward'}, default='savgol'
        Derivative computation method
    savgol_window : int, default=5
        Savitzky-Golay window (must be odd)
    savgol_order : int, default=3
        Savitzky-Golay polynomial order
    """

    def __init__(
        self,
        min_pattern_length: int = 30,
        target_dim: int = 0,
        derivative_method: Literal['savgol', 'central', 'forward'] = 'savgol',
        savgol_window: int = 5,
        savgol_order: int = 3
    ):
        self.min_pattern_length = min_pattern_length
        self.target_dim = target_dim
        self.derivative_method = derivative_method
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order

    def compute_derivative(self, signal: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Compute derivative of a single signal."""
        if self.derivative_method == 'savgol':
            return savgol_filter(
                signal,
                window_length=self.savgol_window,
                polyorder=self.savgol_order,
                deriv=1,
                delta=dt
            )
        elif self.derivative_method == 'central':
            derivative = np.zeros_like(signal)
            derivative[1:-1] = (signal[2:] - signal[:-2]) / (2 * dt)
            derivative[0] = (signal[1] - signal[0]) / dt
            derivative[-1] = (signal[-1] - signal[-2]) / dt
            return derivative
        elif self.derivative_method == 'forward':
            return np.diff(signal, prepend=signal[0]) / dt
        else:
            raise ValueError(f"Unknown derivative method: {self.derivative_method}")

    def compute_multivariate_derivatives(
        self,
        time_series: np.ndarray,
        dt: float = 1.0
    ) -> np.ndarray:
        """
        Compute derivatives for all dimensions.

        Parameters
        ----------
        time_series : np.ndarray, shape (n_timesteps, n_dims)
            Multivariate time series
        dt : float
            Time step

        Returns
        -------
        derivatives : np.ndarray, shape (n_timesteps, n_dims)
            Derivatives for each dimension
        """
        n_timesteps, n_dims = time_series.shape
        derivatives = np.zeros_like(time_series)

        for d in range(n_dims):
            derivatives[:, d] = self.compute_derivative(time_series[:, d], dt)

        return derivatives

    def find_best_derivative_match(
        self,
        derivative_context: np.ndarray,
        derivative_query: np.ndarray
    ) -> Tuple[int, float]:
        """Find best matching derivative pattern across all dimensions."""
        pattern_length, n_dims = derivative_query.shape
        n_context = derivative_context.shape[0]

        best_correlation = -np.inf
        best_idx = 0

        # Normalize query
        query_norm = np.zeros_like(derivative_query)
        for d in range(n_dims):
            query_norm[:, d] = (
                (derivative_query[:, d] - np.mean(derivative_query[:, d])) /
                (np.std(derivative_query[:, d]) + 1e-8)
            )

        # Slide through context
        for i in range(n_context - pattern_length + 1):
            window = derivative_context[i:i + pattern_length, :]

            correlations = []
            for d in range(n_dims):
                window_norm = (
                    (window[:, d] - np.mean(window[:, d])) /
                    (np.std(window[:, d]) + 1e-8)
                )
                corr = np.corrcoef(query_norm[:, d], window_norm)[0, 1]
                correlations.append(corr)

            avg_correlation = np.mean(correlations)

            if avg_correlation > best_correlation:
                best_correlation = avg_correlation
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
        Generate forecast using multivariate derivative parroting.

        Parameters
        ----------
        time_series : np.ndarray, shape (n_timesteps, n_dims)
            Multivariate time series
        horizon : int
            Forecast horizon
        dt : float
            Time step
        pattern_length : int, optional
            Pattern length to match

        Returns
        -------
        forecast : np.ndarray, shape (horizon,)
            Forecast for target dimension
        metadata : dict
            Information about forecast
        """
        if pattern_length is None:
            pattern_length = self.min_pattern_length

        n_timesteps, n_dims = time_series.shape

        if self.target_dim >= n_dims:
            raise ValueError(f"target_dim {self.target_dim} >= n_dims {n_dims}")

        # Compute derivatives for all dimensions
        derivatives = self.compute_multivariate_derivatives(time_series, dt)

        # Extract recent derivative pattern (all dimensions)
        query_derivatives = derivatives[-pattern_length:, :]

        # Search in earlier context
        search_derivatives = derivatives[:-pattern_length, :]

        # Find best match
        match_idx, correlation = self.find_best_derivative_match(
            search_derivatives,
            query_derivatives
        )

        # Extract mimicked derivatives for TARGET dimension only
        continuation_start = match_idx + pattern_length
        continuation_end = min(continuation_start + horizon, len(search_derivatives))
        available_horizon = continuation_end - continuation_start

        mimicked_derivatives = search_derivatives[
            continuation_start:continuation_end,
            self.target_dim
        ].copy()

        # Pad if needed
        if available_horizon < horizon:
            padding = np.full(
                horizon - available_horizon,
                mimicked_derivatives[-1] if len(mimicked_derivatives) > 0 else derivatives[-1, self.target_dim]
            )
            mimicked_derivatives = np.concatenate([mimicked_derivatives, padding])

        # Generate forecast recurrently: y_{t+1} = y_t + d * dt
        forecast = np.zeros(horizon)
        current_value = time_series[-1, self.target_dim]

        for i in range(horizon):
            current_value = current_value + mimicked_derivatives[i] * dt
            forecast[i] = current_value

        metadata = {
            'match_idx': match_idx,
            'pattern_length': pattern_length,
            'multivariate_derivative_correlation': correlation,
            'n_dimensions': n_dims,
            'target_dimension': self.target_dim,
            'derivative_method': self.derivative_method,
            'continuation_available': available_horizon,
            'mean_mimicked_derivative': np.mean(mimicked_derivatives),
            'std_mimicked_derivative': np.std(mimicked_derivatives)
        }

        return forecast, metadata


if __name__ == "__main__":
    print("=" * 70)
    print("MULTIVARIATE PARROTING FORECASTERS")
    print("=" * 70)
    print("\nPattern matching across multiple signals")
    print("Forecasting a single target dimension\n")

    # Generate synthetic multivariate chaotic-like data
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    dt = t[1] - t[0]

    # Three coupled signals (like Lorenz x, y, z)
    x = np.sin(t) + 0.5 * np.sin(3 * t)
    y = np.cos(t) + 0.3 * np.cos(5 * t)
    z = 0.5 * np.sin(2 * t) + 0.2 * np.cos(7 * t)

    # Stack into multivariate series
    time_series = np.column_stack([x, y, z])

    # Add small noise
    time_series += 0.05 * np.random.randn(*time_series.shape)

    # Split data
    split_idx = 700
    context = time_series[:split_idx, :]
    test_data = time_series[split_idx:split_idx + 100, 0]  # Test on dimension 0 (x)

    print(f"Data shape: {context.shape}")
    print(f"Dimensions: {context.shape[1]}")
    print(f"Context length: {context.shape[0]}")
    print(f"Forecast horizon: {len(test_data)}")
    print(f"Target dimension: 0 (first signal)")

    # Test 1: Multivariate Context Parroting
    print("\n" + "=" * 70)
    print("1. MULTIVARIATE CONTEXT PARROTING")
    print("=" * 70)

    mv_context = MultivarateContextParrotForecaster(
        min_pattern_length=30,
        target_dim=0
    )

    forecast_context, meta_context = mv_context.forecast(
        time_series=context,
        horizon=len(test_data),
        pattern_length=50
    )

    from context_parrot_benchmark import evaluate_forecast
    metrics_context = evaluate_forecast(test_data, forecast_context)

    print(f"\nMatch found at index: {meta_context['match_idx']}")
    print(f"Multivariate correlation: {meta_context['multivariate_correlation']:.4f}")
    print(f"Dimensions used: {meta_context['n_dimensions']}")
    print(f"\nPerformance:")
    print(f"  sMAPE: {metrics_context['sMAPE']:.2f}")
    print(f"  MAE: {metrics_context['MAE']:.4f}")
    print(f"  RMSE: {metrics_context['RMSE']:.4f}")

    # Test 2: Multivariate Derivative Parroting
    print("\n" + "=" * 70)
    print("2. MULTIVARIATE DERIVATIVE PARROTING")
    print("=" * 70)

    mv_derivative = MultivariateDerivativeParrotForecaster(
        min_pattern_length=30,
        target_dim=0,
        derivative_method='savgol'
    )

    forecast_deriv, meta_deriv = mv_derivative.forecast(
        time_series=context,
        horizon=len(test_data),
        dt=dt,
        pattern_length=50
    )

    metrics_deriv = evaluate_forecast(test_data, forecast_deriv)

    print(f"\nMatch found at index: {meta_deriv['match_idx']}")
    print(f"Multivariate derivative correlation: {meta_deriv['multivariate_derivative_correlation']:.4f}")
    print(f"Dimensions used: {meta_deriv['n_dimensions']}")
    print(f"Derivative method: {meta_deriv['derivative_method']}")
    print(f"\nPerformance:")
    print(f"  sMAPE: {metrics_deriv['sMAPE']:.2f}")
    print(f"  MAE: {metrics_deriv['MAE']:.4f}")
    print(f"  RMSE: {metrics_deriv['RMSE']:.4f}")

    # Compare with univariate approach
    print("\n" + "=" * 70)
    print("3. COMPARISON: Multivariate vs Univariate")
    print("=" * 70)

    from context_parrot_benchmark import ContextParrotForecaster

    univariate = ContextParrotForecaster(min_pattern_length=30)
    forecast_uni, _ = univariate.forecast(
        time_series=context[:, 0],  # Only use x dimension
        horizon=len(test_data),
        pattern_length=50
    )
    metrics_uni = evaluate_forecast(test_data, forecast_uni)

    print("\nsMAPE Comparison:")
    print(f"  Univariate (x only):         {metrics_uni['sMAPE']:.2f}")
    print(f"  Multivariate (x, y, z):      {metrics_context['sMAPE']:.2f}")
    print(f"  Multivariate Derivative:     {metrics_deriv['sMAPE']:.2f}")

    improvement = ((metrics_uni['sMAPE'] - metrics_context['sMAPE']) / metrics_uni['sMAPE']) * 100
    print(f"\nImprovement: {improvement:.1f}% better using multivariate patterns")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("Multivariate pattern matching uses information from ALL signals")
    print("to find better matches, often improving forecast accuracy.")
