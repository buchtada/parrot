# Parrot Forecast - Files Overview

## Core Algorithm Implementations

### `context_parrot_benchmark.py`
**Context Parroting Algorithm** - Pattern matching in value space

- **Class**: `ContextParrotForecaster`
- **Method**: Finds similar value patterns in historical data
- **Forecasting**: Repeats what came after matching patterns
- **Key Feature**: Simple, interpretable, directly from Zhang & Gilpin paper

### `derivative_parrot_benchmark.py`
**Derivative Parroting Algorithm** - Pattern matching in derivative space

- **Class**: `DerivativeParrotForecaster`
- **Method**: Finds similar derivative (rate of change) patterns
- **Forecasting**: Recurrent application y_{t+1} = y_t + d_mimicked * δt
- **Key Feature**: Three derivative methods (Savgol, Central, Forward) for maximum accuracy
- **When to use**: Better for systems where velocity is more predictable than position

## Visualization & Demos

### `forecast_dashboard.html`
**Professional Web Dashboard** - Interactive time series analysis tool

- **Technology**: Plotly.js for professional charting
- **Design**: Clean black/grey/gold color scheme
- **Features**:
  - Real-time interactive plots
  - Performance metric cards
  - Best method highlighting
  - Responsive layout
  - Production-ready appearance

### `generate_forecast_data.py`
**Data Generator** - Creates JSON for web dashboard

- Runs all forecasting methods
- Generates `forecast_data.json`
- Includes metadata and performance metrics
- Command: `python generate_forecast_data.py`

### `demo_comparison.py`
**Side-by-Side Comparison** - Matplotlib-based comparison (legacy)

- Compares all three methods
- Generates comparison plots
- Command line visualization
- *Note: HTML dashboard is preferred for presentation*

## Data Files

### `Zhang_Gilpin.pdf`
Original research paper from ICLR 2025
- "Zero-shot Forecasting of Chaotic Systems"
- Foundation for context parroting algorithm

### `forecast_data.json` (generated)
JSON output containing:
- Time series data
- All forecast results
- Performance metrics
- Metadata

## Documentation

### `README.md`
Main documentation with:
- Algorithm descriptions
- Usage examples
- Performance comparisons
- Installation instructions

### `FILES_OVERVIEW.md` (this file)
Quick reference guide to all files

## Quick Start Guide

### For Development:
```bash
# Test individual algorithms
python context_parrot_benchmark.py
python derivative_parrot_benchmark.py
```

### For Professional Presentation:
```bash
# Generate data
python generate_forecast_data.py

# Open dashboard
open forecast_dashboard.html
```

### For Research/Analysis:
```python
from context_parrot_benchmark import ContextParrotForecaster
from derivative_parrot_benchmark import DerivativeParrotForecaster

# Use in your own code
forecaster = DerivativeParrotForecaster(derivative_method='savgol')
forecast, metadata = forecaster.forecast(your_data, horizon=100, dt=0.1)
```

## Color Scheme

The professional dashboard uses a sophisticated palette:
- **Background**: Gradient from #1a1a1a to #2d2d2d (dark charcoal)
- **Primary**: #d4af37 (gold) - for highlights and important metrics
- **Secondary**: #e0e0e0 (light grey) - for text
- **Accents**: #808080 (medium grey) - for labels and borders
- **White**: #ffffff - for true data visualization

## Architecture

```
Input Time Series
        ↓
┌───────┴───────┐
│               │
Context         Derivative
Parroting       Parroting
│               │
└───────┬───────┘
        ↓
  Forecasts
        ↓
generate_forecast_data.py
        ↓
  forecast_data.json
        ↓
forecast_dashboard.html
        ↓
  Professional
  Visualization
```

## Performance Notes

- **Context Parroting**: Fast, simple, good baseline (sMAPE ~25)
- **Derivative Savgol**: Smooth derivatives, robust to noise (sMAPE ~23)
- **Derivative Central**: Often best accuracy (sMAPE ~15-18)
- **Choice**: Depends on whether values or velocities are more stable

## Future Enhancements

Potential additions:
- Real-time data streaming
- More derivative methods (spectral, polynomial)
- Multi-step ahead forecasting with uncertainty
- Integration with live data sources
- Export to different formats
- Automated model selection
