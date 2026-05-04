import yaml
from pathlib import Path
import json
import joblib
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import learning_curve

def load_config(path):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    config_path = project_root / path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def save_experiment(model_name,
                    pipeline,
                    metrics,
                    config,
                    X,
                    y,
                    scoring="f1_macro",
                    cv=5,
                    feature_importance_df=None,
                    confusion_matrix=None
                    ):
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

    if feature_importance_df is not None:
        feature_importance_df.to_csv(exp_dir / "feature_importance.csv")

    if confusion_matrix is not None:
        confusion_matrix.to_csv(exp_dir / "confusion_matrix.csv")

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X,
        y,
        cv=cv,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        scoring=scoring
        )
    
    lc_df = pd.DataFrame({
        "train_size": train_sizes,
        "train_score_mean": train_scores.mean(axis=1),
        "train_score_std": train_scores.std(axis=1),
        "val_score_mean": val_scores.mean(axis=1),
        "val_score_std": val_scores.std(axis=1)
    })

    lc_df.to_csv(exp_dir / "learning_curve.csv", index=False)

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


def load_feature_importances():
    root = Path(__file__).resolve().parents[2]
    model_root = root / "models"

    results = []

    for model_dir in model_root.iterdir():
        fi_file = model_dir / "feature_importance.csv"
        config_file = model_dir / "config.yaml"

        if fi_file.exists():
            fi_df = pd.read_csv(fi_file)

            if config_file.exists():
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)

            for _, row in fi_df.iterrows():
                results.append({
                    "model": config["model"],
                    "feature": row["feature"],
                    "coefficient": row["coefficient"]
                })

    return pd.DataFrame(results)


def build_model_colors(models, palette):
    palette = sns.color_palette(palette, n_colors=len(models))
    return {model: palette[i] for i, model in enumerate(models)}


def plot_radar_chart(df, metric_cols, figsize, model_colors, title):

    n_metrics = len(metric_cols)

    angles = [n / float(n_metrics) * 2 * np.pi for n in range(n_metrics)]
    angles += angles[:1]

    plt.figure(figsize=figsize)

    for _, row in df.iterrows():
        values = row[metric_cols].tolist()
        values += values[:1]

        plt.polar(
            angles,
            values,
            color=model_colors[row["model"]],
            linewidth=2,
            label=row["model"]
        )

    ax = plt.gca()
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    pretty_labels = [col.replace("macro_", "").replace("_", " ").title() for col in metric_cols]

    for angle, label in zip(angles[:-1], pretty_labels):
        angle_deg = np.degrees(angle)
        r_max = ax.get_ylim()[1]

        ha = "left" if -90 <= angle_deg <= 90 else "right"
        va = "bottom" if 0 < angle_deg < 180 else "top"

        ax.text(angle, r_max * 1.05, label, ha=ha, va=va)

    plt.legend(loc='lower right', bbox_to_anchor=(1.2, -0.05))
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(cmap):
    root = Path(__file__).resolve().parents[2]
    model_root = root / "models"

    model_dirs = []
    titles = []

    for d in model_root.iterdir():
        cm_file = d / "confusion_matrix.csv"
        config_file = d / "config.yaml"

        if d.is_dir() and cm_file.exists() and config_file.exists():
            model_dirs.append(d)

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            titles.append(config["model"])

    n = len(model_dirs)

    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]

            if idx < n:
                model_dir = model_dirs[idx]
                title = titles[idx]

                df_cm = pd.read_csv(model_dir / "confusion_matrix.csv", index_col=0)

                sns.heatmap(df_cm, annot=True, cmap=cmap, ax=ax, vmin=0, vmax=1)
                ax.set_title(title)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
            else:
                ax.axis("off")

            idx += 1

    plt.suptitle("Confusion Matrices")
    plt.tight_layout()
    plt.show()