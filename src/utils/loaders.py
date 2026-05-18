from pathlib import Path
import yaml
import pandas as pd
import os


def load_config(path):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    config_path = project_root / path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

def load_metrics(repo):
    results = []

    for model_dir, config in repo.iter_models():
        metrics = repo.load_json(model_dir, "metrics.json")
        if metrics is None:
            continue

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


def load_class_metrics(repo):
    results = []

    for model_dir, config in repo.iter_models():
        metrics = repo.load_json(model_dir, "metrics.json")
        if metrics is None:
            continue

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


def load_feature_importances(repo):
    results = []

    for model_dir, config in repo.iter_models():
        fi_df = repo.load_csv(model_dir, "feature_importance.csv")
        if fi_df is None:
            continue

        for _, row in fi_df.iterrows():
            results.append({
                "model": config["model"],
                "feature": row["feature"],
                "coefficient": row["coefficient"]
                })

    return pd.DataFrame(results)


def load_confusion_data(repo):
    results = []

    for model_dir, config in repo.iter_models():
        cm_df = repo.load_csv(model_dir, "confusion_matrix.csv")
        cm_df = cm_df.set_index("Unnamed: 0")
        
        if cm_df is not None:
            results.append((config["model"], cm_df))
    
    return results


def load_learning_curves(repo):
    results = []

    for model_dir, config in repo.iter_models():
        lc_df = repo.load_csv(model_dir, "learning_curve.csv")

        if lc_df is not None:
            results.append((config["model"], lc_df))

    return results


def load_misclassifications(repo):
    results = []

    for model_dir, config in repo.iter_models():
        mscls_df = repo.load_csv(model_dir, "misclassifications.csv")

        if mscls_df is not None:
            results.append((config["model"], mscls_df))

    return results

def load_model_sizes(repo):
    results = []

    for model_dir, config in repo.iter_models():
        model_size = os.path.getsize(model_dir / "model.pkl")

        if model_size is not None:
            results.append({"model": config["model"], "model_size_kb": model_size / 1024})
    return pd.DataFrame(results)
