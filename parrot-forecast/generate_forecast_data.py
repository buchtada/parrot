"""
Generate forecast data for HTML visualization
Creates JSON output for professional web-based time series dashboard
"""

import numpy as np
import json
from context_parrot_benchmark import ContextParrotForecaster, evaluate_forecast
from derivative_parrot_benchmark import DerivativeParrotForecaster


def generate_chaotic_signal(length=1000, seed=42):
    """Generate a chaotic-like time series for testing."""
    np.random.seed(seed)
    t = np.linspace(0, 100, length)
    dt = t[1] - t[0]

    signal = (
        np.sin(t) +
        0.5 * np.sin(3 * t) +
        0.2 * np.cos(5 * t) +
        0.1 * np.sin(7 * t)
    )
    signal += 0.05 * np.random.randn(length)

    return t, signal, dt


def run_all_forecasts():
    """Run all forecasting methods and return results as JSON-serializable dict."""

    # Generate data
    t, signal, dt = generate_chaotic_signal(length=1000)

    # Split data
    split_idx = 700
    test_horizon = 100

    context_data = signal[:split_idx]
    test_data = signal[split_idx:split_idx + test_horizon]

    t_context = t[:split_idx]
    t_test = t[split_idx:split_idx + test_horizon]

    # Show last 150 points of context for visualization
    viz_start = max(0, len(context_data) - 150)

    results = {
        'metadata': {
            'dt': float(dt),
            'context_length': len(context_data),
            'test_horizon': len(test_data),
            'pattern_length': 50
        },
        'time': {
            'context': t_context[viz_start:].tolist(),
            'test': t_test.tolist()
        },
        'data': {
            'context': context_data[viz_start:].tolist(),
            'test': test_data.tolist()
        },
        'forecasts': {}
    }

    # 1. Context Parroting
    print("Running Context Parroting...")
    context_forecaster = ContextParrotForecaster(min_pattern_length=30)
    forecast_context, meta_context = context_forecaster.forecast(
        time_series=context_data,
        horizon=len(test_data),
        pattern_length=50
    )
    metrics_context = evaluate_forecast(test_data, forecast_context)

    results['forecasts']['context_parroting'] = {
        'name': 'Context Parroting',
        'description': 'Matches value patterns directly',
        'forecast': forecast_context.tolist(),
        'metrics': {k: float(v) for k, v in metrics_context.items()},
        'metadata': {
            'match_idx': int(meta_context['match_idx']),
            'correlation': float(meta_context['correlation'])
        }
    }

    # 2. Derivative Parroting - Savgol
    print("Running Derivative Parroting (Savitzky-Golay)...")
    deriv_savgol = DerivativeParrotForecaster(
        min_pattern_length=30,
        derivative_method='savgol'
    )
    forecast_savgol, meta_savgol = deriv_savgol.forecast(
        time_series=context_data,
        horizon=len(test_data),
        dt=dt,
        pattern_length=50
    )
    metrics_savgol = evaluate_forecast(test_data, forecast_savgol)

    results['forecasts']['derivative_savgol'] = {
        'name': 'Derivative Parroting (Savgol)',
        'description': 'Mimics smooth derivative patterns',
        'forecast': forecast_savgol.tolist(),
        'metrics': {k: float(v) for k, v in metrics_savgol.items()},
        'metadata': {
            'match_idx': int(meta_savgol['match_idx']),
            'derivative_correlation': float(meta_savgol['derivative_correlation']),
            'method': meta_savgol['derivative_method']
        }
    }

    # 3. Derivative Parroting - Central
    print("Running Derivative Parroting (Central Differences)...")
    deriv_central = DerivativeParrotForecaster(
        min_pattern_length=30,
        derivative_method='central'
    )
    forecast_central, meta_central = deriv_central.forecast(
        time_series=context_data,
        horizon=len(test_data),
        dt=dt,
        pattern_length=50
    )
    metrics_central = evaluate_forecast(test_data, forecast_central)

    results['forecasts']['derivative_central'] = {
        'name': 'Derivative Parroting (Central)',
        'description': 'Mimics exact derivative patterns',
        'forecast': forecast_central.tolist(),
        'metrics': {k: float(v) for k, v in metrics_central.items()},
        'metadata': {
            'match_idx': int(meta_central['match_idx']),
            'derivative_correlation': float(meta_central['derivative_correlation']),
            'method': meta_central['derivative_method']
        }
    }

    return results


if __name__ == "__main__":
    print("Generating forecast data for web visualization...")
    print("=" * 60)

    results = run_all_forecasts()

    # Save to JSON
    output_path = 'forecast_data.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nData saved to: {output_path}")
    print(f"Generated {len(results['forecasts'])} forecasts")

    print("\nPerformance Summary:")
    print("-" * 60)
    for key, forecast in results['forecasts'].items():
        print(f"{forecast['name']:40s} sMAPE: {forecast['metrics']['sMAPE']:6.2f}")

    print("\nReady for HTML visualization!")
