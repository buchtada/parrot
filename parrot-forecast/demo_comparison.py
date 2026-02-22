"""
Side-by-side comparison of Context Parroting vs Derivative Parroting

This demo shows how both algorithms perform on the same time series,
illustrating when each approach excels.
"""

import numpy as np
import matplotlib.pyplot as plt
from context_parrot_benchmark import ContextParrotForecaster, evaluate_forecast
from derivative_parrot_benchmark import DerivativeParrotForecaster


def generate_chaotic_signal(length=1000, seed=42):
    """Generate a chaotic-like time series for testing."""
    np.random.seed(seed)
    t = np.linspace(0, 100, length)
    dt = t[1] - t[0]

    # Combine multiple frequencies for chaotic-like behavior
    signal = (
        np.sin(t) +
        0.5 * np.sin(3 * t) +
        0.2 * np.cos(5 * t) +
        0.1 * np.sin(7 * t)
    )
    signal += 0.05 * np.random.randn(length)  # Add noise

    return t, signal, dt


def compare_forecasters(context_data, test_data, dt, pattern_length=50):
    """Compare all forecasting methods."""

    results = {}

    # 1. Context Parroting
    print("\n1. Context Parroting")
    print("-" * 40)
    context_forecaster = ContextParrotForecaster(min_pattern_length=30)
    forecast_context, meta_context = context_forecaster.forecast(
        time_series=context_data,
        horizon=len(test_data),
        pattern_length=pattern_length
    )
    metrics_context = evaluate_forecast(test_data, forecast_context)

    print(f"  Match correlation: {meta_context['correlation']:.4f}")
    print(f"  sMAPE: {metrics_context['sMAPE']:.2f}")
    print(f"  MAE: {metrics_context['MAE']:.4f}")

    results['context'] = {
        'forecast': forecast_context,
        'metadata': meta_context,
        'metrics': metrics_context
    }

    # 2. Derivative Parroting - Savitzky-Golay
    print("\n2. Derivative Parroting (Savitzky-Golay)")
    print("-" * 40)
    deriv_savgol = DerivativeParrotForecaster(
        min_pattern_length=30,
        derivative_method='savgol'
    )
    forecast_savgol, meta_savgol = deriv_savgol.forecast(
        time_series=context_data,
        horizon=len(test_data),
        dt=dt,
        pattern_length=pattern_length
    )
    metrics_savgol = evaluate_forecast(test_data, forecast_savgol)

    print(f"  Derivative correlation: {meta_savgol['derivative_correlation']:.4f}")
    print(f"  sMAPE: {metrics_savgol['sMAPE']:.2f}")
    print(f"  MAE: {metrics_savgol['MAE']:.4f}")

    results['savgol'] = {
        'forecast': forecast_savgol,
        'metadata': meta_savgol,
        'metrics': metrics_savgol
    }

    # 3. Derivative Parroting - Central Differences
    print("\n3. Derivative Parroting (Central Differences)")
    print("-" * 40)
    deriv_central = DerivativeParrotForecaster(
        min_pattern_length=30,
        derivative_method='central'
    )
    forecast_central, meta_central = deriv_central.forecast(
        time_series=context_data,
        horizon=len(test_data),
        dt=dt,
        pattern_length=pattern_length
    )
    metrics_central = evaluate_forecast(test_data, forecast_central)

    print(f"  Derivative correlation: {meta_central['derivative_correlation']:.4f}")
    print(f"  sMAPE: {metrics_central['sMAPE']:.2f}")
    print(f"  MAE: {metrics_central['MAE']:.4f}")

    results['central'] = {
        'forecast': forecast_central,
        'metadata': meta_central,
        'metrics': metrics_central
    }

    return results


def plot_comparison(t_context, context_data, t_test, test_data, results, save_path=None):
    """Create a visualization comparing all methods."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Parrot Forecasting Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Context Parroting
    ax1 = axes[0, 0]
    ax1.plot(t_context[-100:], context_data[-100:], 'k-', alpha=0.5, label='Context', linewidth=2)
    ax1.plot(t_test, test_data, 'b-', label='True', linewidth=2)
    ax1.plot(t_test, results['context']['forecast'], 'r--', label='Forecast', linewidth=2)
    ax1.set_title('Context Parroting\n(Pattern Value Matching)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f"sMAPE: {results['context']['metrics']['sMAPE']:.2f}",
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Derivative Parroting (Savgol)
    ax2 = axes[0, 1]
    ax2.plot(t_context[-100:], context_data[-100:], 'k-', alpha=0.5, label='Context', linewidth=2)
    ax2.plot(t_test, test_data, 'b-', label='True', linewidth=2)
    ax2.plot(t_test, results['savgol']['forecast'], 'g--', label='Forecast', linewidth=2)
    ax2.set_title('Derivative Parroting (Savgol)\n(Smooth Derivatives)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f"sMAPE: {results['savgol']['metrics']['sMAPE']:.2f}",
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Plot 3: Derivative Parroting (Central)
    ax3 = axes[1, 0]
    ax3.plot(t_context[-100:], context_data[-100:], 'k-', alpha=0.5, label='Context', linewidth=2)
    ax3.plot(t_test, test_data, 'b-', label='True', linewidth=2)
    ax3.plot(t_test, results['central']['forecast'], 'm--', label='Forecast', linewidth=2)
    ax3.set_title('Derivative Parroting (Central Diff)\n(Exact Derivatives)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, f"sMAPE: {results['central']['metrics']['sMAPE']:.2f}",
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))

    # Plot 4: Performance Summary
    ax4 = axes[1, 1]
    methods = ['Context\nParroting', 'Derivative\n(Savgol)', 'Derivative\n(Central)']
    smapes = [
        results['context']['metrics']['sMAPE'],
        results['savgol']['metrics']['sMAPE'],
        results['central']['metrics']['sMAPE']
    ]
    colors = ['red', 'green', 'magenta']

    bars = ax4.bar(methods, smapes, color=colors, alpha=0.6, edgecolor='black', linewidth=2)
    ax4.set_ylabel('sMAPE (lower is better)', fontweight='bold')
    ax4.set_title('Forecast Accuracy Comparison', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, smape in zip(bars, smapes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{smape:.1f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("PARROT FORECASTING: SIDE-BY-SIDE COMPARISON")
    print("=" * 60)

    # Generate data
    print("\nGenerating chaotic-like time series...")
    t, signal, dt = generate_chaotic_signal(length=1000)

    # Split data
    split_idx = 700
    test_horizon = 100

    context_data = signal[:split_idx]
    test_data = signal[split_idx:split_idx + test_horizon]

    t_context = t[:split_idx]
    t_test = t[split_idx:split_idx + test_horizon]

    print(f"Context length: {len(context_data)}")
    print(f"Test horizon: {len(test_data)}")
    print(f"Time step (dt): {dt:.4f}")

    # Compare all methods
    print("\n" + "=" * 60)
    print("FORECASTING RESULTS")
    print("=" * 60)

    results = compare_forecasters(context_data, test_data, dt, pattern_length=50)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_method = min(results.keys(), key=lambda k: results[k]['metrics']['sMAPE'])
    print(f"\nBest performing method: {best_method.upper()}")
    print(f"  sMAPE: {results[best_method]['metrics']['sMAPE']:.2f}")
    print(f"  MAE: {results[best_method]['metrics']['MAE']:.4f}")

    print("\nKey insights:")
    print("  • Context parroting: Matches value patterns directly")
    print("  • Derivative parroting: Matches rate-of-change patterns")
    print("  • Central differences often provide best derivative accuracy")
    print("  • Choice depends on whether values or velocities are more predictable")

    # Create visualization
    try:
        print("\nGenerating comparison plot...")
        fig = plot_comparison(t_context, context_data, t_test, test_data, results,
                             save_path='parrot_forecast_comparison.png')
        print("Plot created successfully!")
        print("\nNote: Close the plot window to exit.")
        plt.show()
    except Exception as e:
        print(f"Could not create plot: {e}")
        print("(matplotlib may not be available)")
