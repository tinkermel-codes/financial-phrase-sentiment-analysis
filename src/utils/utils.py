import yaml
from pathlib import Path
import os
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

def load_config(path):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    config_path = project_root / path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def save_experiment(model_name, pipeline, metrics, config, feature_importance_fig=None):
    root = Path(__file__).resolve().parents[2]
    models_dir = root / "models"

    exp_name = model_name.lower().replace(" ", "_")
    exp_dir = models_dir / f"{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, exp_dir / "model.pkl")

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    if feature_importance_fig is not None:
        feature_importance_fig.savefig(exp_dir / "feature_importance.png", dpi=150)

    print(f"Experiment saved to: {exp_dir}")


def load_metrics():
    root = Path(__file__).resolve().parents[2]
    model_root = root / "models"

    results = []

    for model_dir in model_root.iterdir():
        metrics_file = model_dir / "metrics.json"
        config_file = model_dir / "config.yaml"

        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

        results.append({
            "model": config["model"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro avg"]["f1-score"],
            "weighted_f1": metrics["weighted avg"]["f1-score"],
            "macro_precision": metrics["macro avg"]["precision"],
            "macro_recall": metrics["macro avg"]["recall"],
            "config": config
            })

    return pd.DataFrame(results)


def load_class_metrics():
    root = Path(__file__).resolve().parents[2]
    model_root = root / "models"

    results = []

    for model_dir in model_root.iterdir():
        metrics_file = model_dir / "metrics.json"
        config_file = model_dir / "config.yaml"

        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

        for cls, values in metrics.items():
            if cls in ["accuracy", "macro avg", "weighted avg"]:
                continue

            results.append({
                "model": config["model"],
                "class": cls,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": values["support"]
            })

    return pd.DataFrame(results)
