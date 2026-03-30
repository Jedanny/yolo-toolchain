"""
Hyperparameter Tuner Module

Wraps YOLO's built-in hyperparameter evolution (genetic algorithm) to automatically
search for optimal hyperparameters.

Features:
- Genetic algorithm hyperparameter search
- Custom parameter space definition
- Early stopping based on metric improvement
- Tuning results comparison
- Visualization of tuning progress
"""

import csv
import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

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

    # Space parameters - define which parameters to tune and their ranges
    # Format: {param_name: (min, max)} or {param_name: [list_of_values]}
    space: Optional[Dict[str, Any]] = None

    # Early stopping
    patience: int = 50  # Stop if no improvement for N generations

    # Early stopping metric
    metric: str = "metrics/mAP50(B)"  # Metric to optimize
    direction: str = "maximize"  # "maximize" or "minimize"

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
        # Add space parameters if specified
        if self.space:
            result["space"] = self.space
        # Merge tune_kwargs
        result.update(self.tune_kwargs)
        # Remove None values
        return {k: v for k, v in result.items() if v is not None}

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


class HyperparameterTuner:
    """Hyperparameter tuner using YOLO's built-in genetic algorithm."""

    # Default parameter space for YOLO
    DEFAULT_SPACE = {
        'lr0': (0.001, 0.1),      # Initial learning rate
        'lrf': (0.01, 1.0),        # Final learning rate factor
        'momentum': (0.6, 0.98),    # SGD momentum
        'weight_decay': (0.0, 0.001),  # Weight decay
        'warmup_epochs': (0.0, 5.0),  # Warmup epochs
        'box': (0.02, 0.2),         # Box loss gain
        'cls': (0.2, 2.0),          # cls loss gain
        'dfl': (0.5, 2.0),          # DFL loss gain
        'hsv_h': (0.0, 0.1),        # Hue augmentation
        'hsv_s': (0.0, 0.9),        # Saturation augmentation
        'hsv_v': (0.0, 0.9),        # Value augmentation
        'degrees': (0.0, 45.0),      # Rotation
        'translate': (0.0, 0.5),     # Translation
        'scale': (0.0, 0.9),        # Scale
        'shear': (0.0, 10.0),       # Shear
        'perspective': (0.0, 0.001), # Perspective
        'flipud': (0.0, 1.0),       # Vertical flip
        'fliplr': (0.0, 1.0),       # Horizontal flip
        'mosaic': (0.0, 1.0),       # Mosaic
        'mixup': (0.0, 1.0),        # MixUp
        'copy_paste': (0.0, 1.0),   # Copy-paste
    }

    def __init__(self, config: TunerConfig):
        """Initialize the tuner with configuration.

        Args:
            config: TunerConfig instance with tuning parameters
        """
        self.config = config
        self.model = None
        self.results = {}
        self.tuning_history = []
        self.best_metric = float('-inf') if config.direction == 'maximize' else float('inf')
        self.patience_counter = 0

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

        # Save config
        self.config.to_yaml(str(output_path / "tuner_config.yaml"))

        # Run tuning
        tune_params = self.config.to_dict()

        # Use model.tune() with genetic algorithm
        self.model.tune(**tune_params)

        # Load best results from output
        best_result = self._load_best_from_output()

        # Extract evolution history
        evolution_history = self._extract_evolution_history()

        # Analyze tuning results
        analysis = self._analyze_results(evolution_history)

        # Generate comparison table
        comparison = self._generate_comparison(evolution_history)

        # Save results
        final_results = {
            "best": best_result,
            "evolution_history": evolution_history,
            "analysis": analysis,
            "comparison": comparison,
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        self._save_results(final_results)

        # Generate visualizations
        self._generate_visualizations(evolution_history, output_path)

        return final_results

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

    def _analyze_results(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze tuning results and generate insights.

        Args:
            history: Evolution history from tuning

        Returns:
            Analysis dictionary with insights
        """
        if not history:
            return {}

        # Extract metrics columns
        metric_cols = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        param_cols = [k for k in history[0].keys() if k not in metric_cols and k not in ['epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'lr/pg0', 'fitness']]

        analysis = {
            'generations': len(history),
            'total_trials': len(history),
            'metrics_summary': {},
            'param_importance': {},
        }

        # Calculate metrics summary
        for col in metric_cols:
            values = []
            for row in history:
                if col in row and row[col]:
                    try:
                        values.append(float(row[col]))
                    except (ValueError, TypeError):
                        pass
            if values:
                analysis['metrics_summary'][col] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': self._calculate_std(values),
                }

        # Calculate parameter importance (correlation with fitness)
        fitness_values = []
        for row in history:
            if 'fitness' in row and row['fitness']:
                try:
                    fitness_values.append(float(row['fitness']))
                except (ValueError, TypeError):
                    pass

        if fitness_values and len(fitness_values) == len(history):
            for param in param_cols:
                param_values = []
                for row in history:
                    if param in row and row[param]:
                        try:
                            param_values.append(float(row[param]))
                        except (ValueError, TypeError):
                            pass
                if param_values and len(param_values) == len(fitness_values):
                    corr = self._calculate_correlation(param_values, fitness_values)
                    analysis['param_importance'][param] = round(corr, 4)

        # Sort by importance
        analysis['param_importance'] = dict(
            sorted(analysis['param_importance'].items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return analysis

    def _generate_comparison(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate comparison table of top tuning runs.

        Args:
            history: Evolution history

        Returns:
            List of top runs sorted by fitness
        """
        if not history:
            return []

        # Filter rows with fitness
        rows_with_fitness = []
        for row in history:
            if 'fitness' in row and row['fitness']:
                try:
                    row['fitness_float'] = float(row['fitness'])
                    rows_with_fitness.append(row)
                except (ValueError, TypeError):
                    pass

        # Sort by fitness
        rows_with_fitness.sort(key=lambda x: x['fitness_float'], reverse=True)

        # Take top 5
        top_rows = []
        for i, row in enumerate(rows_with_fitness[:5]):
            top_row = {
                'rank': i + 1,
                'fitness': row.get('fitness', 'N/A'),
            }
            # Add key parameters
            key_params = ['lr0', 'lrf', 'momentum', 'box', 'cls', 'dfl', 'mosaic', 'mixup']
            for param in key_params:
                if param in row:
                    top_row[param] = row[param]
            # Add key metrics
            for col in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']:
                if col in row:
                    top_row[col.split('/')[1].replace('(B)', '')] = row[col]
            top_rows.append(top_row)

        return top_rows

    def _generate_visualizations(self, history: List[Dict[str, Any]], output_path: Path) -> None:
        """Generate visualization plots for tuning results.

        Args:
            history: Evolution history
            output_path: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Warning: matplotlib not installed, skipping visualization")
            return

        if not history:
            return

        # Extract data
        epochs = []
        fitness = []
        map50 = []
        map50_95 = []

        for row in history:
            if 'epoch' in row and 'fitness' in row:
                try:
                    epochs.append(int(row['epoch']))
                    fitness.append(float(row['fitness']))
                    if 'metrics/mAP50(B)' in row and row['metrics/mAP50(B)']:
                        map50.append(float(row['metrics/mAP50(B)']))
                    else:
                        map50.append(0)
                    if 'metrics/mAP50-95(B)' in row and row['metrics/mAP50-95(B)']:
                        map50_95.append(float(row['metrics/mAP50-95(B)']))
                    else:
                        map50_95.append(0)
                except (ValueError, TypeError):
                    pass

        if not epochs:
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=14)

        # Plot 1: Fitness over epochs
        ax1 = axes[0, 0]
        ax1.plot(epochs, fitness, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness over Training')
        ax1.grid(True, alpha=0.3)
        if fitness:
            ax1.axhline(y=max(fitness), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(fitness):.4f}')
            ax1.legend()

        # Plot 2: mAP50 over epochs
        ax2 = axes[0, 1]
        ax2.plot(epochs, map50, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP50')
        ax2.set_title('mAP50 over Training')
        ax2.grid(True, alpha=0.3)
        if map50:
            ax2.axhline(y=max(map50), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(map50):.4f}')
            ax2.legend()

        # Plot 3: mAP50-95 over epochs
        ax3 = axes[1, 0]
        ax3.plot(epochs, map50_95, 'm-', linewidth=2, marker='^', markersize=4)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('mAP50-95')
        ax3.set_title('mAP50-95 over Training')
        ax3.grid(True, alpha=0.3)
        if map50_95:
            ax3.axhline(y=max(map50_95), color='r', linestyle='--', alpha=0.5, label=f'Best: {max(map50_95):.4f}')
            ax3.legend()

        # Plot 4: Convergence - fitness vs mAP50
        ax4 = axes[1, 1]
        ax4.scatter(map50, fitness, alpha=0.6, c=epochs, cmap='viridis', s=50)
        ax4.set_xlabel('mAP50')
        ax4.set_ylabel('Fitness')
        ax4.set_title('mAP50 vs Fitness (color = epoch)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'tuning_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to: {output_path / 'tuning_visualization.png'}")

    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    @staticmethod
    def _calculate_correlation(x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denom_x = (sum((x[i] - mean_x) ** 2 for i in range(n))) ** 0.5
        denom_y = (sum((y[i] - mean_y) ** 2 for i in range(n))) ** 0.5
        if denom_x * denom_y == 0:
            return 0.0
        return numerator / (denom_x * denom_y)

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
