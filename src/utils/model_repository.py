from pathlib import Path
import yaml
import json
import pandas as pd
import joblib

class ModelRepository:
    def __init__(self, root = None):
        self.root = root or Path(__file__).resolve().parents[2]
        self.model_root = self.root / "models"


    def get_experiment_dir(self, model_name):
        exp_name = model_name.lower().replace(" ", "_")
        exp_dir = self.model_root / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    
    def save_model(self, path, pipeline):
        joblib.dump(pipeline, path)


    def save_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


    def save_yaml(self, path, data):
        with open(path, "w") as f:
            yaml.safe_dump(data, f)
    

    def save_dataframe(self, path, df, index=True):
        df.to_csv(path, index=index)


    def save_experiment(self,
                        model_name,
                        pipeline,
                        metrics,
                        config,
                        feature_importance_df=None,
                        confusion_matrix=None,
                        learning_curve_df=None,
                        misclassiifactions_df=None
                        ):
        exp_dir = self.get_experiment_dir(model_name)

        self.save_model(exp_dir / "model.pkl", pipeline)
        self.save_json(exp_dir / "metrics.json", metrics)
        self.save_yaml(exp_dir / "config.yaml", config)

        if feature_importance_df is not None:
            self.save_dataframe(exp_dir / "feature_importance.csv", feature_importance_df)

        if confusion_matrix is not None:
            self.save_dataframe(exp_dir / "confusion_matrix.csv", confusion_matrix)

        if learning_curve_df is not None:
            self.save_dataframe(exp_dir / "learning_curve.csv", learning_curve_df, index=False)

        if misclassiifactions_df is not None:
            self.save_dataframe(exp_dir / "misclassifications.csv", misclassiifactions_df, index=False)

        print(f"Experiment saved to: {exp_dir}")

        return exp_dir


    def iter_models(self):
        for model_dir in self.model_root.iterdir():
            config_file = model_dir / "config.yaml"
            if not config_file.exists():
                continue
            
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            yield model_dir, config


    def load_json(self, model_dir, filename):
        file = model_dir / filename
        
        if file.exists():
            with open(file, "r") as f:
                return json.load(f)
        return None
    

    def load_csv(self, model_dir, filename):
        file = model_dir / filename

        if file.exists():
            return pd.read_csv(file)
        return None