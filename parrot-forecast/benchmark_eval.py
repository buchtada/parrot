"""
Benchmark Evaluation Script for Parrot Forecasters

Evaluates the parrot forecasting models against:
1. FEV-Bench (autogluon/fev)
2. GIFT-Eval (Salesforce)

Run this script to get local benchmark results before submission.
"""

import subprocess
import sys

def install_dependencies():
    """Install required benchmark packages."""
    packages = [
        "fev",  # FEV-Bench
        "datasets",  # Hugging Face datasets
        "gluonts",  # Time series utilities
        "torch",  # Required by many models
        "pandas",
        "tqdm",
    ]

    print("Installing benchmark dependencies...")
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
            print(f"  {pkg}")
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Failed to install {pkg}: {e}")

    print("Dependencies installed.\n")

if __name__ == "__main__":
    print("=" * 70)
    print("PARROT FORECASTER BENCHMARK EVALUATION")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Install required dependencies")
    print("2. Run FEV-Bench evaluation")
    print("3. Run GIFT-Eval evaluation")
    print("4. Generate comparison report")
    print("\n")

    install_dependencies()

    print("Running evaluations...")
    print("See fev_bench_eval.py and gift_eval_runner.py for individual benchmarks.")
