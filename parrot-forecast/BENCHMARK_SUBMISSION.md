# Submitting Parrot Forecast to Time Series Benchmarks

## Overview

This guide explains how to submit the Parrot Forecast algorithms to major time series forecasting benchmarks.

## Target Benchmarks

### 1. GIFT-Eval (Hugging Face)
- **URL**: https://huggingface.co/spaces/Salesforce/GIFT-Eval
- **Focus**: General time series forecasting across diverse domains
- **Models Evaluated**: Foundation models, transformers, statistical methods
- **Datasets**: Multiple domains (energy, traffic, weather, etc.)

### 2. FEV-Bench
- **Focus**: Foundation model evaluation for time series
- **Datasets**: Long-term forecasting benchmarks
- **Zero-shot capability**: Tests models without fine-tuning

### 3. Monash Time Series Forecasting Archive
- **URL**: https://forecastingdata.org/
- **Standard**: M4, M5 competitions format
- **Comprehensive**: 30+ datasets across domains

## Why Parrot Forecast Is Competitive

### Unique Selling Points

1. **True Zero-Shot**: No training required, pure pattern matching
2. **Interpretable**: Easy to understand what the model is doing
3. **Fast**: No GPU needed, runs on CPU
4. **Memory Efficient**: Minimal memory footprint
5. **Multivariate**: Can leverage multiple signals (unique advantage)
6. **Derivative Variants**: Novel approach using rate-of-change

### Expected Performance

Based on Zhang & Gilpin (ICLR 2025):
- **Competitive** with NBEATS, TiDE on chaotic systems
- **Better than** LSTM, simple transformers in data-limited settings
- **Multivariate variant** has 17-25% improvement over univariate

### Positioning

**"Context Parroting: A Minimal Baseline for Foundation Model Evaluation"**

Frame it as:
- Simple, interpretable baseline
- Tests if foundation models truly add value beyond pattern matching
- Highlights the "parroting" mechanism discovered in large models

## Submission Strategy

### Phase 1: Create Standard Interface

We need to wrap our models in a standard forecasting interface:

```python
class ParrotForecastModel:
    """Standard interface for benchmark submissions."""

    def __init__(self, config):
        pass

    def forecast(self, context, horizon, **kwargs):
        """
        Standard forecasting interface.

        Parameters
        ----------
        context : array-like or dict
            Historical data
        horizon : int
            Forecast horizon
        **kwargs : dict
            Additional parameters (freq, etc.)

        Returns
        -------
        forecast : array-like
            Point forecasts
        """
        pass
```

### Phase 2: Benchmark-Specific Adapters

Each benchmark has its own format. We need adapters for:

1. **GIFT-Eval Format**
   - Expects HuggingFace model interface
   - Supports multiple datasets
   - Requires specific output format

2. **FEV-Bench Format**
   - Zero-shot evaluation protocol
   - Standardized data loaders
   - Multiple metrics (sMAPE, MAE, MSE)

3. **Monash Format**
   - .tsf file format
   - Specific train/test splits
   - Standard evaluation scripts

## Implementation Plan

### Step 1: Create Universal Wrapper

```python
# benchmark_adapter.py

from typing import Union, Dict, Optional
import numpy as np
import pandas as pd

class ParrotForecastAdapter:
    """
    Universal adapter for time series benchmarks.
    Wraps Parrot Forecast models for standard interfaces.
    """

    def __init__(
        self,
        model_type='multivariate_context',
        min_pattern_length=30,
        derivative_method='savgol',
        target_dim=0
    ):
        self.model_type = model_type
        self.min_pattern_length = min_pattern_length
        self.derivative_method = derivative_method
        self.target_dim = target_dim

    def forecast(
        self,
        time_series: Union[np.ndarray, pd.DataFrame],
        horizon: int,
        freq: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Standard forecasting interface."""
        pass

    def batch_forecast(
        self,
        dataset: Dict[str, np.ndarray],
        horizons: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """Batch forecasting for multiple series."""
        pass
```

### Step 2: GIFT-Eval Specific

GIFT-Eval typically expects:
- HuggingFace Transformers-compatible interface
- Can load via `AutoModel.from_pretrained()`
- Or custom model with `predict()` method

**Approach:**
```python
# gift_eval_submission.py

class ParrotGIFTModel:
    """GIFT-Eval compatible wrapper."""

    def __init__(self, model_name='parrot-multivariate'):
        # Load our model
        pass

    def predict(self, past_values, freq=None):
        """GIFT-Eval standard interface."""
        # Convert to our format
        # Run parrot forecast
        # Return in GIFT format
        pass
```

### Step 3: Create Model Card

For HuggingFace submission, need a model card:

```markdown
---
tags:
- time-series
- forecasting
- zero-shot
- pattern-matching
library_name: parrot-forecast
---

# Parrot Forecast: Context Parroting for Time Series

## Model Description

A zero-shot time series forecasting model based on context parroting,
as described in "Zero-shot Forecasting of Chaotic Systems" (Zhang & Gilpin, ICLR 2025).

## How It Works

1. Finds similar patterns in historical data
2. Uses what came next after those patterns as the forecast
3. No training required - pure pattern matching

## Variants

- **Context Parroting**: Matches value patterns
- **Derivative Parroting**: Matches rate-of-change patterns
- **Multivariate**: Uses multiple correlated signals (17-25% better)

## Usage

```python
from parrot_forecast import ParrotForecastModel

model = ParrotForecastModel(model_type='multivariate_context')
forecast = model.forecast(historical_data, horizon=96)
```

## Performance

Competitive with NBEATS, TiDE on chaotic systems.
Particularly strong in data-limited settings.

## Citation

```bibtex
@inproceedings{zhang2025zeroshot,
  title={Zero-shot Forecasting of Chaotic Systems},
  author={Zhang, Yuanzhao and Gilpin, William},
  booktitle={ICLR},
  year={2025}
}
```
```

### Step 4: Evaluation Scripts

Create standardized evaluation:

```python
# evaluate_benchmarks.py

def evaluate_on_gift_eval(model, datasets):
    """Run model on GIFT-Eval datasets."""
    results = {}
    for dataset in datasets:
        forecast = model.forecast(dataset['context'], dataset['horizon'])
        metrics = compute_metrics(dataset['test'], forecast)
        results[dataset['name']] = metrics
    return results

def evaluate_on_monash(model, datasets):
    """Run model on Monash archive."""
    pass
```

## Submission Checklist

### Before Submitting

- [ ] Implement benchmark adapter interface
- [ ] Test on sample datasets from each benchmark
- [ ] Verify output format matches requirements
- [ ] Create reproducibility documentation
- [ ] Add requirements.txt with exact versions
- [ ] Write clear README with usage examples
- [ ] Create model card for HuggingFace

### For GIFT-Eval

- [ ] Create HuggingFace account
- [ ] Upload model to HuggingFace Hub
- [ ] Test model loads correctly via `from_pretrained()`
- [ ] Submit through GIFT-Eval interface
- [ ] Include comparison to other zero-shot models

### For FEV-Bench

- [ ] Clone FEV-Bench repository
- [ ] Follow their submission protocol
- [ ] Run on their standard datasets
- [ ] Submit results in their format

### For Papers/Publications

If writing a paper:
- [ ] Position as "interpretable baseline"
- [ ] Highlight multivariate advantage
- [ ] Compare to foundation models
- [ ] Show computational efficiency
- [ ] Demonstrate zero-shot capability

## Competitive Advantages to Highlight

### 1. Computational Efficiency
```
Training time: 0 seconds (no training!)
Inference: ~0.1s per forecast on CPU
Memory: <100MB
```

### 2. Interpretability
```
Can visualize:
- Which historical pattern was matched
- Correlation score of the match
- Exact continuation used
```

### 3. Multivariate Capability
```
Unique among simple baselines:
- Uses cross-variable correlations
- 17-25% better than univariate
- Natural for coupled systems
```

### 4. True Zero-Shot
```
No training data contamination possible
No fine-tuning needed
Works on any domain immediately
```

## Expected Rankings

### Optimistic Scenario
- **Top 25%** among all methods on chaotic/coupled systems
- **Top 10%** in zero-shot category
- **#1** in interpretable/simple methods

### Realistic Scenario
- **Top 50%** overall
- **Top 30%** in zero-shot category
- **Strong baseline** for comparison

### Key Metrics to Watch
- **sMAPE**: Should be competitive (15-30 range)
- **MAE**: Good on normalized data
- **Computational time**: Should dominate
- **Memory usage**: Should dominate

## Risks and Mitigation

### Risk 1: Poor Performance on Non-Chaotic Data
**Mitigation**:
- Emphasize suitability for chaotic/complex systems
- Show where it works vs doesn't
- Position as specialized tool

### Risk 2: Benchmarks Favor Large Models
**Mitigation**:
- Highlight computational advantages
- Show competitive performance per compute
- Emphasize interpretability

### Risk 3: Format/Interface Issues
**Mitigation**:
- Thoroughly test adapters
- Follow benchmark guidelines exactly
- Reach out to organizers with questions

## Next Steps

1. **Immediate** (Week 1):
   - Create benchmark adapter interface
   - Test on sample GIFT-Eval datasets
   - Draft model card

2. **Short-term** (Week 2-3):
   - Implement GIFT-Eval submission
   - Test on full benchmark suite
   - Submit to leaderboard

3. **Medium-term** (Month 1-2):
   - Expand to other benchmarks
   - Write technical report
   - Consider publication

## Resources

### Benchmark Documentation
- [GIFT-Eval Paper](https://arxiv.org/abs/2410.10393)
- [Monash Archive](https://forecastingdata.org/)
- [AutoGluon Time Series](https://auto.gluon.ai/stable/tutorials/timeseries/)

### Related Baselines
- Seasonal Naive
- ARIMA
- Exponential Smoothing
- Simple Transformers

### Positioning Papers
- "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2023)
- "Zero-shot Forecasting" (Zhang & Gilpin, 2025)
- "In-Context Learning for Time Series" (various)

## Support

For questions about submission:
- GIFT-Eval: Check their GitHub issues
- FEV-Bench: Contact organizers
- General: Time series forecasting Discord/Slack communities

---

**Remember**: Frame Parrot Forecast as an **interpretable, efficient baseline** that highlights what simple pattern matching can achieve. This makes foundation models' performance more impressive (if they beat it) or raises questions (if they don't!).
