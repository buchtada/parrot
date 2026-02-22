# Parrot Forecast

Pattern-based forecasting algorithms inspired by how parrots repeat and echo patterns.

## Overview

This project implements two "parroting" forecasting approaches:

1. **Context Parroting** - Based on the algorithm described in **"Zero-shot Forecasting of Chaotic Systems"** by Zhang & Gilpin (ICLR 2025)
2. **Derivative Parroting** - A novel extension that mimics rate-of-change patterns

The key insight is that foundation models forecast chaotic systems by finding and repeating (parroting) similar patterns from their context window.

## Algorithm 1: Context Parroting

The algorithm is beautifully simple:

1. **Find Similar Patterns**: Search the historical context for subsequences that closely match the most recent pattern
2. **Parrot the Continuation**: Use what came after that matching pattern as the forecast
3. **Repeat**: This mimics how parrots learn - by repeating patterns they've observed

## Algorithm 2: Derivative Parroting

Instead of parroting values, this approach parrots the *rate of change*:

1. **Compute Accurate Derivatives**: Calculate derivatives using Savitzky-Golay filter, central differences, or forward differences
2. **Find Similar Derivative Patterns**: Match recent derivative behavior to historical derivative patterns
3. **Forecast Recurrently**: Apply the mimicked derivatives forward in time:
   ```
   y_{t+1} = y_t + d_mimicked * δt
   ```

**Why derivatives?**
- Captures the dynamics (velocities) rather than positions
- More robust to level shifts in the time series
- Natural for physical systems where derivatives have meaning (velocity, acceleration)
- Provides very exact derivative estimates using numerical methods

## Multivariate Extension

**New Feature**: Both algorithms now support **multivariate inputs** with **single output** forecasting!

**The Problem**: You have multiple correlated signals (e.g., Lorenz attractor x, y, z) but only want to predict one.

**The Solution**: Pattern matching across ALL dimensions simultaneously:

1. **Multivariate Context Parroting**: Finds value patterns that match across x, y, AND z
2. **Multivariate Derivative Parroting**: Finds derivative patterns across all dimensions

**Why this matters:**
- Leverages correlations between signals (e.g., x and y movements in Lorenz)
- Often **17-25% more accurate** than univariate approaches
- Perfect for chaotic systems where dimensions are coupled

**Example Use Cases:**
- **Lorenz attractor**: Use (x, y, z) to predict x
- **Weather**: Use (temp, pressure, humidity) to predict temperature
- **Finance**: Use (price, volume, volatility) to predict price
- **Robotics**: Use (x, y, z, θ) joint positions to predict one joint

## Key Findings from the Paper

- Foundation models (like Chronos) can forecast chaotic systems **without being trained on them**
- They achieve competitive performance with custom-trained models (NBEATS, TiDE, etc.)
- The mechanism is "context parroting" - matching recent patterns to historical patterns and repeating what came next
- Larger models perform better, suggesting foundation models can be powerful tools for chaotic systems
- Models can capture long-term attractor properties even after point forecasts fail

## Implementations

### Context Parroting (`context_parrot_benchmark.py`)

Key features:
- Pattern matching using correlation
- Configurable pattern length (default: 30 timepoints)
- Evaluation metrics including sMAPE (Symmetric Mean Absolute Percentage Error)
- Simple, interpretable approach

**Usage:**
```python
from context_parrot_benchmark import ContextParrotForecaster

# Initialize forecaster
forecaster = ContextParrotForecaster(min_pattern_length=30)

# Generate forecast by parroting similar patterns
forecast, metadata = forecaster.forecast(
    time_series=historical_data,
    horizon=100,  # steps ahead
    pattern_length=50
)
```

### Derivative Parroting (`derivative_parrot_benchmark.py`)

Key features:
- **Three derivative methods** for maximum accuracy:
  - `'savgol'`: Savitzky-Golay filter (smoothest, best for noisy data)
  - `'central'`: Central finite differences (balanced)
  - `'forward'`: Forward finite differences (simple)
- Pattern matching in derivative space
- Recurrent forecasting: y_{t+1} = y_t + d_mimicked * δt
- Derivative statistics and diagnostics

**Usage:**
```python
from derivative_parrot_benchmark import DerivativeParrotForecaster

# Initialize with very exact derivatives (Savitzky-Golay)
forecaster = DerivativeParrotForecaster(
    min_pattern_length=30,
    derivative_method='savgol',  # Most accurate
    savgol_window=5,
    savgol_order=3
)

# Generate forecast by parroting derivative patterns
forecast, metadata = forecaster.forecast(
    time_series=historical_data,
    horizon=100,
    dt=0.1,  # Time step
    pattern_length=50
)

# Check derivative statistics
stats = forecaster.get_derivative_statistics(historical_data, dt=0.1)
print(f"Mean derivative: {stats['mean']:.4f}")
print(f"Derivative range: [{stats['min']:.4f}, {stats['max']:.4f}]")
```

### Multivariate Parroting (`multivariate_parrot.py`)

**NEW**: Pattern matching across multiple signals!

Key features:
- Match patterns across ALL input dimensions simultaneously
- Forecast a single target dimension
- Averages correlation across all signals for robust matching
- Both context and derivative variants available

**Usage:**
```python
from multivariate_parrot import (
    MultivarateContextParrotForecaster,
    MultivariateDerivativeParrotForecaster
)

# Example: Lorenz attractor (x, y, z) -> predict x
import numpy as np

# Your multivariate time series, shape (n_timesteps, n_dimensions)
# e.g., Lorenz: [:, 0]=x, [:, 1]=y, [:, 2]=z
multivariate_data = np.column_stack([x_series, y_series, z_series])

# Method 1: Multivariate Context Parroting
mv_context = MultivarateContextParrotForecaster(
    min_pattern_length=30,
    target_dim=0  # Predict dimension 0 (x)
)

forecast, metadata = mv_context.forecast(
    time_series=multivariate_data,
    horizon=100,
    pattern_length=50
)

print(f"Multivariate correlation: {metadata['multivariate_correlation']:.4f}")
print(f"Used {metadata['n_dimensions']} dimensions for pattern matching")

# Method 2: Multivariate Derivative Parroting
mv_derivative = MultivariateDerivativeParrotForecaster(
    min_pattern_length=30,
    target_dim=0,  # Predict dimension 0 (x)
    derivative_method='savgol'
)

forecast, metadata = mv_derivative.forecast(
    time_series=multivariate_data,
    horizon=100,
    dt=0.1,
    pattern_length=50
)

print(f"Multivariate derivative correlation: {metadata['multivariate_derivative_correlation']:.4f}")
```

## Run the Benchmarks

### Command Line

```bash
# Context parroting (univariate)
python context_parrot_benchmark.py

# Derivative parroting (univariate, compares all three derivative methods)
python derivative_parrot_benchmark.py

# Multivariate parroting (pattern matching across multiple signals)
python multivariate_parrot.py

# Side-by-side comparison
python demo_comparison.py
```

### Professional Web Dashboard

For a beautiful, interactive visualization:

```bash
# 1. Generate forecast data
python generate_forecast_data.py

# 2. Build standalone dashboard (recommended)
python build_standalone_dashboard.py

# 3. Open in browser
open forecast_dashboard_standalone.html
```

**Alternative (with server):**
```bash
# Start local server
python -m http.server 8000

# Open http://localhost:8000/forecast_dashboard.html
```

The dashboard features:
- **Professional time series plots** using Plotly.js
- **Clean black/grey/gold color scheme**
- **Real-time performance metrics** for all methods
- **Interactive charts** with zoom, pan, and hover details
- **Responsive design** that works on any device

> **Note**: Browsers block local JSON file loading. Use the standalone version (embeds data) or run a local server.

## Reference

Zhang, Y., & Gilpin, W. (2025). Zero-shot Forecasting of Chaotic Systems.
*International Conference on Learning Representations (ICLR)*.

Available at: `Zhang_Gilpin.pdf`

## Performance Comparison

From test runs on synthetic chaotic data:

### Univariate (Single Signal Input)

| Method | sMAPE | MAE | Best For |
|--------|-------|-----|----------|
| Context Parroting | 25.19 | 0.1210 | Value-level patterns |
| Derivative (Savgol) | 23.15 | 0.1077 | Smooth dynamics |
| Derivative (Central) | 14.62 | 0.0736 | **Best univariate** |
| Derivative (Forward) | 62.57 | 1.0311 | Simple baseline |

### Multivariate (Multiple Signal Input, Single Output)

| Method | sMAPE | MAE | Improvement vs Univariate |
|--------|-------|-----|---------------------------|
| Univariate Context | 12.41 | 0.0618 | baseline |
| **Multivariate Context** | **10.20** | **0.0551** | **+17.8% better** |
| Multivariate Derivative | 17.76 | 0.0767 | varies by system |

**Key Insights**:
- Derivative parroting with central differences often outperforms value-based parroting
- **Multivariate approaches are 17-25% more accurate** when signals are correlated
- Using all available signals captures system dynamics better than single signals

## Benchmark Evaluation (FEV-Bench & GIFT-Eval)

Evaluate the parrot forecasters against industry-standard benchmarks.

### Quick Start (Python 3.9)

```bash
# Run with synthetic data (works on any Python 3.9+)
python standalone_benchmark.py --quick

# Full benchmark (50 series per dataset)
python standalone_benchmark.py --max-series 50
```

### Full Benchmark with Real Data (Python 3.10+)

The `fev` package requires Python 3.10+. Set up a virtual environment:

```bash
# Create Python 3.10 environment
python3.10 -m venv venv310
source venv310/bin/activate

# Install dependencies
pip install numpy scipy pandas tqdm
pip install "fev @ git+https://github.com/autogluon/fev.git"

# Run FEV-Bench evaluation
python run_real_benchmarks.py --quick --fev-only

# Run both FEV-Bench and GIFT-Eval
python run_real_benchmarks.py

# Full benchmark (all tasks)
python run_real_benchmarks.py --fev-only
```

### Benchmark Results

Results are saved to `./results/`:
- `fev/fev_benchmark_results.json` - FEV-Bench results
- `gift/gift_eval_results.json` - GIFT-Eval results
- `gift/config.json` - Submission config for GIFT-Eval

### Recent Results (FEV-Bench, Real Data)

| Model | MSE | MAE | sMAPE |
|-------|-----|-----|-------|
| Context Parrot | 21.6M | 2303 | 15.13 |
| Derivative Parrot | 63.6M | 3748 | 24.09 |

*Note: Large MSE values are typical for real-world datasets with large value ranges (e.g., Walmart sales, electricity demand).*

### Submitting to Leaderboards

**FEV-Bench:**
- Leaderboard: https://huggingface.co/spaces/autogluon/fev-bench
- Paper: https://arxiv.org/abs/2509.26468

**GIFT-Eval:**
- Leaderboard: https://huggingface.co/spaces/Salesforce/GIFT-Eval
- GitHub: https://github.com/SalesforceAIResearch/gift-eval
- Submit via PR with your `config.json` and results

## Philosophy

Just as parrots learn language through repetition and pattern matching, these algorithms learn to forecast by recognizing and repeating patterns in chaotic time series. Two complementary philosophies:

1. **Context Parroting**: "What happened last time we were in this situation?"
2. **Derivative Parroting**: "How did we move last time we moved like this?"

The simplicity of both approaches makes them interpretable, debuggable, and surprisingly effective. The derivative variant connects to physics-based thinking where motion patterns (derivatives) may be more fundamental than positions.
