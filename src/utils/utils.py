import yaml
from pathlib import Path
import os
import json
import joblib
import matplotlib.pyplot as plt


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
    exp_dir = models_dir / f"experiment_{exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, exp_dir / "model.pkl")

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    if feature_importance_fig is not None:
        feature_importance_fig.savefig(exp_dir / "feature_importance.png", dpi=150)

    print(f"Experiment saved to: {exp_dir}")