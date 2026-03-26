"""
Hyperparameter Tuner Module

Wraps YOLO's built-in hyperparameter evolution (genetic algorithm) to automatically
search for optimal hyperparameters.
"""

import os
import csv
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

from ultralytics import YOLO


@dataclass
class TunerConfig:
    """Configuration for hyperparameter tuning."""

    # Model
    model: str = "yolo11n.pt"

    # Dataset
    data: str = "data.yaml"

    # Evolution settings
    epochs: int = 100
    iterations: int = 10  # Number of generations for genetic algorithm

    # Output
    output_dir: str = "runs/tune"

    # Storage for evolution results
    storage: str = "sqlite:///:memory:"

    # Resume tuning
    resume: bool = False

    # Space parameters (optional, YOLO uses defaults if not specified)
    space: Optional[Dict[str, Any]] = None

    # Early stopping
    patience: int = 50

    # Device
    device: str = ""

    # Additional YOLO tune kwargs
    tune_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for YOLO tune()."""
        result = {
            "data": self.data,
            "epochs": self.epochs,
            "iterations": self.iterations,
            "storage": self.storage,
            "resume": self.resume,
            "patience": self.patience,
            "device": self.device if self.device else None,
        }
        # Merge tune_kwargs
        result.update(self.tune_kwargs)
        # Remove None values
        return {k: v for k, v in result.items() if v is not None}


class HyperparameterTuner:
    """Hyperparameter tuner using YOLO's built-in genetic algorithm."""

    def __init__(self, config: TunerConfig):
        """Initialize the tuner with configuration.

        Args:
            config: TunerConfig instance with tuning parameters
        """
        self.config = config
        self.model = None
        self.results = {}

    def tune(self) -> Dict[str, Any]:
        """Run hyperparameter tuning using genetic algorithm.

        Returns:
            Dictionary containing best hyperparameters and metrics
        """
        # Load model
        self.model = YOLO(self.config.model)

        # Create output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run tuning
        tune_params = self.config.to_dict()

        # Use model.tune() with genetic algorithm
        self.model.tune(**tune_params)

        # Load best results from output
        best_result = self._load_best_from_output()

        # Extract evolution history
        evolution_history = self._extract_evolution_history()

        # Save results
        self._save_results({
            "best": best_result,
            "evolution_history": evolution_history,
            "config": self.config.to_dict(),
        })

        return self.results

    def _load_best_from_output(self) -> Dict[str, Any]:
        """Load best hyperparameters from YOLO's output directory.

        Returns:
            Dictionary with best hyperparameters found
        """
        output_path = Path(self.config.output_dir)

        # YOLO outputs best results to runs/tune/weights/best.pt
        # and creates a results.csv with evolution data
        best = {}

        # Find the tune directory (YOLO creates timestamped directories)
        tune_dirs = list(output_path.glob("tune*"))
        if not tune_dirs:
            return best

        # Use most recent tune directory
        latest_tune = max(tune_dirs, key=lambda p: p.stat().st_mtime)
        results_csv = latest_tune / "results.csv"

        if results_csv.exists():
            # Read the last row (best result) from results.csv
            with open(results_csv, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    # Get the row with best fitness (last column typically)
                    best_row = rows[-1]
                    best = {k: v for k, v in best_row.items() if v}

        # Also check for best.pt weights
        weights_dir = latest_tune / "weights"
        if weights_dir.exists():
            best_weights = list(weights_dir.glob("best.pt"))
            if best_weights:
                best["best_weights"] = str(best_weights[0])

        best["output_dir"] = str(latest_tune)
        return best

    def _extract_evolution_history(self) -> List[Dict[str, Any]]:
        """Extract evolution history from CSV files.

        Returns:
            List of dictionaries containing evolution history
        """
        output_path = Path(self.config.output_dir)
        history = []

        # Find tune directories
        tune_dirs = list(output_path.glob("tune*"))
        if not tune_dirs:
            return history

        latest_tune = max(tune_dirs, key=lambda p: p.stat().st_mtime)
        results_csv = latest_tune / "results.csv"

        if results_csv.exists():
            with open(results_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    history.append({k: v for k, v in row.items() if v})

        return history

    def _save_results(self, result: Dict[str, Any]) -> None:
        """Save tuning results to YAML file.

        Args:
            result: Dictionary containing tuning results
        """
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / "tuning_results.yaml"

        with open(results_file, "w") as f:
            yaml.dump(result, f, default_flow_style=False, sort_keys=False)

        self.results = result

    @staticmethod
    def load_config_from_yaml(yaml_path: str) -> TunerConfig:
        """Load TunerConfig from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            TunerConfig instance
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Handle nested config structure
        if "tuner" in data:
            data = data["tuner"]

        # Extract tune_kwargs if present
        tune_kwargs = data.pop("tune_kwargs", {})

        config = TunerConfig(**data)
        config.tune_kwargs = tune_kwargs

        return config


def main():
    """CLI entry point for hyperparameter tuner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter tuner using YOLO's genetic algorithm"
    )

    # Model and data
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt", help="Path to model weights"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to dataset YAML"
    )

    # Evolution settings
    parser.add_argument(
        "--epochs", type=int, default=100, help="Epochs per generation"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of generations (iterations)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/tune",
        help="Output directory for tuning results",
    )

    # Storage
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///:memory:",
        help="Storage URL for evolution results",
    )

    # Resume
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last tuning run"
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="", help="Device to use (e.g., '0' or 'cpu')"
    )

    # Config file
    parser.add_argument(
        "--config", type=str, help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    # Load config from file or use arguments
    if args.config:
        config = HyperparameterTuner.load_config_from_yaml(args.config)
    else:
        config = TunerConfig(
            model=args.model,
            data=args.data,
            epochs=args.epochs,
            iterations=args.iterations,
            output_dir=args.output_dir,
            storage=args.storage,
            resume=args.resume,
            device=args.device,
        )

    # Run tuning
    tuner = HyperparameterTuner(config)
    results = tuner.tune()

    print("\n" + "=" * 60)
    print("Hyperparameter Tuning Complete!")
    print("=" * 60)
    print(f"\nBest results saved to: {config.output_dir}/tuning_results.yaml")
    print(f"Best weights: {results.get('best', {}).get('best_weights', 'N/A')}")


if __name__ == "__main__":
    main()
