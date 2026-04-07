import yaml
from pathlib import Path


def load_config(path):
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    config_path = project_root / path
    with open(config_path, "r") as f:
        return yaml.safe_load(f)