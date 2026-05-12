import yaml
from pathlib import Path
import joblib
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def build_model_colors(models, palette):
    palette = sns.color_palette(palette, n_colors=len(models))
    return {model: palette[i] for i, model in enumerate(models)}


def plot_macro_metrics(df, colors, figsize=(10,6)):
    df_melted = df.melt(
        id_vars="model",
        value_vars=["macro_precision", "macro_recall", "macro_f1"],
        var_name="metric",
        value_name="score"
        )

    df_melted["metric"] = (df_melted["metric"].str.replace("_", " ").str.title())

    plt.figure(figsize=figsize)
    sns.barplot(data=df_melted, x="model", y="score", hue="metric", palette=colors)
    plt.title("Macro Metrics per Model")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.show()


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


def plot_class_performances(df, colors, figsize=(12,6)):
    df_melted = df.melt(
        id_vars=["model", "class"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score"
        )

    plt.figure(figsize=figsize)
    sns.barplot(data=df_melted[df_melted["metric"] == "f1"], x="class", y="score", hue="model", palette=colors)
    plt.title("Class-Level Performance per Model")
    plt.xlabel("Class")
    plt.ylabel("F1 Score")
    plt.ylim(0,1.1)
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


def plot_all_learning_curves():
    root = Path(__file__).resolve().parents[2]
    model_root = root / "models"

    model_dirs = []
    titles = []

    for d in model_root.iterdir():
        lc_file = d / "learning_curve.csv"
        config_file = d / "config.yaml"

        if d.is_dir() and lc_file.exists() and config_file.exists():
            model_dirs.append(d)

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            titles.append(config["model"])

    n = len(model_dirs)

    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, model_dir in zip(axes, model_dirs):
        lc_file = model_dir / "learning_curve.csv"
        config_file = model_dir / "config.yaml"

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        model_name = config.get("model", model_dir.name)

        df = pd.read_csv(lc_file)

        ax.plot(df["train_size"], df["train_score_mean"], label="Train")
        ax.plot(df["train_size"], df["val_score_mean"], linestyle="--", label="Val")

        ax.set_title(model_name)
        ax.set_xlabel("Training Samples")
        ax.set_ylabel("Macro F1 Score")
        ax.legend(loc= "lower right")
        ax.set_ylim(0,1)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_misclassifications():
    root = Path(__file__).resolve().parents[2]
    model_root = root / "models"

    config = load_config(root / "configs/config.yaml")

    data_path = config["data"]["preprocessed_path"]
    text_col = config["data"]["text_column"]
    label_col = config["data"]["label_column"]

    df = pd.read_csv(data_path, keep_default_na=False)
    X = df[text_col]
    y = df[label_col]

    misclassifications = {}

    for d in model_root.iterdir():
        model_file = d / "model.pkl"
        config_file = d / "config.yaml"

        if d.is_dir() and model_file.exists() and config_file.exists():

            model = joblib.load(model_file)
            y_pred = model.predict(X)

            df_mis = pd.DataFrame({
            "text": X,
            "true": y,
            "pred": y_pred
            })
            df_mis = df_mis[df_mis["true"] != df_mis["pred"]]

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            misclassifications[config["model"]] = df_mis
    return misclassifications


def plot_misclassifications(mis_dict, normalize=True, same_scale=True, cmap="Reds"):
    models = list(mis_dict.keys())

    n = len(models)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    if same_scale:
        if normalize:
            vmin, vmax = 0, 1
        else:
            vmin = 0
            vmax = max(pd.crosstab(df["true"], df["pred"]).values.max() for df in mis_dict.values())
    else:
        vmin = vmax = None

    for ax, model in zip(axes, models):
        
        if normalize:
            error_matrix = pd.crosstab(mis_dict[model]["true"], mis_dict[model]["pred"], normalize="index")

        else:
            error_matrix = pd.crosstab(mis_dict[model]["true"], mis_dict[model]["pred"])

        sns.heatmap(
            error_matrix,
            annot=True,
            cmap=cmap,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            fmt="d" if not normalize else ".2f"
            )
        
        ax.set_title(model)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Misclassification Heatmaps")
    plt.tight_layout()
    plt.show()


def plot_error_rates(mis_dict, colors, figsize=(8,5)):
    root = Path(__file__).resolve().parents[2]

    config = load_config(root / "configs/config.yaml")

    data_path = config["data"]["preprocessed_path"]
    label_col = config["data"]["label_column"]

    df = pd.read_csv(data_path, keep_default_na=False)
    class_counts = df[label_col].value_counts()

    rows = []

    for model_name, mis_df in mis_dict.items():
        error_counts = mis_df["true"].value_counts()
        error_rate = (error_counts / class_counts).fillna(0)

        for cls, rate in error_rate.items():
            rows.append({"model": model_name, "class":cls, "error_rate": rate})

    errors_df = pd.DataFrame(rows)

    plt.figure(figsize=figsize)
    sns.barplot(
        data=errors_df,
        x="class",
        y="error_rate",
        hue="model",
        palette=colors
    )

    plt.title("Class-wise Error Rates per Model")
    plt.xlabel("Class")
    plt.ylabel("Error Rate")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()


def get_model_sizes():
    root = Path(__file__).resolve().parents[2]
    model_root = root / "models"

    rows = []

    for d in model_root.iterdir():
        model_file = d / "model.pkl"
        config_file = d / "config.yaml"

        if d.is_dir() and model_file.exists() and config_file.exists():
            model_size = os.path.getsize(model_file)

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            rows.append({"model": config["model"], "model_size_kb": model_size / 1024})
    
    return pd.DataFrame(rows)


def plot_model_sizes_and_efficiencies(metrics_df, sizes_df, metric, colors, figsize=(14,5)):
    df = metrics_df.merge(sizes_df, on="model")
    df["efficiency"] = df[metric] / df["model_size_kb"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    sns.scatterplot(
        data=df,
        x="model_size_kb",
        y=metric,
        hue="model",
        palette=colors,
        ax=axes[0]
        )
    
    axes[0].set_title("Model Size vs Performance")
    axes[0].set_xlabel("Model Size (KB)")
    axes[0].set_ylabel(metric)

    sns.barplot(
        data=df,
        x="model",
        y="efficiency",
        hue="model",
        palette=colors,
        ax=axes[1]
    )
    axes[1].set_title("Model Efficiency: Performance per KB")
    axes[1].set_ylabel(f"Efficiency ({metric} / KB)")
    axes[1].set_xlabel("Model")

    plt.tight_layout()
    plt.show()
   