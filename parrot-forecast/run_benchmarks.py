#!/usr/bin/env python3
"""
Run Both FEV-Bench and GIFT-Eval Benchmarks

This is the main entry point for evaluating parrot forecasters
against both benchmarks locally before submission.

Usage:
    python run_benchmarks.py                    # Run both benchmarks
    python run_benchmarks.py --fev              # Run FEV-Bench only
    python run_benchmarks.py --gift             # Run GIFT-Eval only
    python run_benchmarks.py --quick            # Quick test run
    python run_benchmarks.py --full             # Full benchmark run
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def install_dependencies():
    """Install all required dependencies."""
    print("Installing dependencies...")

    packages = [
        "numpy",
        "scipy",
        "pandas",
        "tqdm",
        "datasets",  # Hugging Face datasets
        "fev",  # FEV-Bench
    ]

    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

    print("Dependencies ready.\n")


def run_fev_benchmark(quick: bool = False):
    """Run FEV-Bench evaluation."""
    print("\n" + "=" * 70)
    print("RUNNING FEV-BENCH EVALUATION")
    print("=" * 70)

    from fev_bench_eval import run_fev_benchmark as fev_eval

    num_tasks = 5 if quick else None
    model_types = ["context", "derivative"]

    results = fev_eval(
        model_types=model_types,
        num_tasks=num_tasks,
        save_results=True,
        output_dir="./results/fev"
    )

    return results


def run_gift_benchmark(quick: bool = False):
    """Run GIFT-Eval benchmark."""
    print("\n" + "=" * 70)
    print("RUNNING GIFT-EVAL BENCHMARK")
    print("=" * 70)

    from gift_eval_runner import run_gift_eval_benchmark as gift_eval

    max_series = 20 if quick else 100
    prediction_lengths = [24] if quick else [24, 48, 96]

    results = gift_eval(
        model_types=["context", "derivative"],
        prediction_lengths=prediction_lengths,
        max_series=max_series,
        save_results=True,
        output_dir="./results/gift"
    )

    return results


def generate_comparison_report(fev_results, gift_results, output_path: str = "./results"):
    """Generate a comparison report across both benchmarks."""

    report = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {
            "fev_bench": {},
            "gift_eval": {}
        },
        "summary": {}
    }

    # Process FEV results
    if fev_results:
        for model, data in fev_results.items():
            report["benchmarks"]["fev_bench"][model] = data.get("average_metrics", {})

    # Process GIFT results
    if gift_results:
        for model, results in gift_results.items():
            if results:
                avg_metrics = {}
                for r in results:
                    for metric, value in r.metrics.items():
                        if metric not in avg_metrics:
                            avg_metrics[metric] = []
                        avg_metrics[metric].append(value)

                report["benchmarks"]["gift_eval"][model] = {
                    k: sum(v) / len(v) for k, v in avg_metrics.items()
                }

    # Generate summary
    models_tested = set()
    if fev_results:
        models_tested.update(fev_results.keys())
    if gift_results:
        models_tested.update(gift_results.keys())

    report["summary"]["models_evaluated"] = list(models_tested)
    report["summary"]["benchmarks_run"] = []
    if fev_results:
        report["summary"]["benchmarks_run"].append("FEV-Bench")
    if gift_results:
        report["summary"]["benchmarks_run"].append("GIFT-Eval")

    # Save report
    output = Path(output_path)
    output.mkdir(exist_ok=True, parents=True)

    with open(output / "benchmark_comparison.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\nModels Evaluated: {', '.join(models_tested)}")
    print(f"Benchmarks Run: {', '.join(report['summary']['benchmarks_run'])}")

    if fev_results:
        print("\nFEV-Bench Results:")
        for model, metrics in report["benchmarks"]["fev_bench"].items():
            print(f"\n  {model}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")

    if gift_results:
        print("\nGIFT-Eval Results:")
        for model, metrics in report["benchmarks"]["gift_eval"].items():
            print(f"\n  {model}:")
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")

    print(f"\nFull report saved to: {output / 'benchmark_comparison.json'}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run FEV-Bench and GIFT-Eval benchmarks for Parrot Forecasters"
    )
    parser.add_argument("--fev", action="store_true", help="Run FEV-Bench only")
    parser.add_argument("--gift", action="store_true", help="Run GIFT-Eval only")
    parser.add_argument("--quick", action="store_true", help="Quick test run (subset of data)")
    parser.add_argument("--full", action="store_true", help="Full benchmark run")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")

    args = parser.parse_args()

    print("=" * 70)
    print("PARROT FORECASTER BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Install dependencies
    if not args.skip_install:
        install_dependencies()

    # Determine what to run
    run_fev = args.fev or (not args.fev and not args.gift)
    run_gift = args.gift or (not args.fev and not args.gift)
    quick = args.quick and not args.full

    if quick:
        print("\nRunning in QUICK mode (subset of data for testing)")
    else:
        print("\nRunning FULL benchmark evaluation")

    fev_results = None
    gift_results = None

    # Run benchmarks
    if run_fev:
        try:
            fev_results = run_fev_benchmark(quick=quick)
        except Exception as e:
            print(f"FEV-Bench error: {e}")
            import traceback
            traceback.print_exc()

    if run_gift:
        try:
            gift_results = run_gift_benchmark(quick=quick)
        except Exception as e:
            print(f"GIFT-Eval error: {e}")
            import traceback
            traceback.print_exc()

    # Generate comparison report
    if fev_results or gift_results:
        generate_comparison_report(fev_results, gift_results)

    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review results in ./results/")
    print("2. If results look good, prepare submission:")
    print("   - FEV-Bench: Submit to https://huggingface.co/spaces/autogluon/fev-bench")
    print("   - GIFT-Eval: Submit PR to https://github.com/SalesforceAIResearch/gift-eval")


if __name__ == "__main__":
    main()
