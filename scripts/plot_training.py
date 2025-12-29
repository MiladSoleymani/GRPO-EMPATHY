#!/usr/bin/env python3
"""
Plot training metrics from GRPO Empathy training.

Usage:
    python scripts/plot_training.py --output-dir outputs
    python scripts/plot_training.py --output-dir outputs --save-only
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpro_empathy.utils.plotting import (
    plot_from_output_dir,
    plot_reward_distribution,
    load_training_logs,
    extract_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Plot GRPO Empathy training metrics")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Path to training output directory",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the plot (default: output_dir/training_metrics.png)",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save plot without displaying",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Also plot reward distributions",
    )

    args = parser.parse_args()

    print(f"Loading training logs from: {args.output_dir}")

    # Load and plot metrics
    logs = load_training_logs(args.output_dir)

    if not logs:
        print(f"No training logs found in {args.output_dir}")
        print("Make sure training has completed and logs are saved.")
        print("\nLooking for trainer_state.json...")

        # Check what files exist
        output_path = Path(args.output_dir)
        if output_path.exists():
            print(f"Files in {args.output_dir}:")
            for f in output_path.iterdir():
                print(f"  - {f.name}")
        return 1

    print(f"Found {len(logs)} log entries")

    metrics = extract_metrics(logs)
    print(f"Extracted metrics for {len(metrics['step'])} training steps")

    # Plot main metrics
    save_path = args.save_path
    if save_path is None:
        save_path = str(Path(args.output_dir) / "training_metrics.png")

    plot_from_output_dir(
        args.output_dir,
        save_path=save_path,
        show=not args.save_only,
    )

    # Plot distributions if requested
    if args.distribution:
        dist_path = str(Path(args.output_dir) / "reward_distributions.png")
        plot_reward_distribution(
            metrics,
            save_path=dist_path,
            show=not args.save_only,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
