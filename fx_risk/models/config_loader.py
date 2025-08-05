"""
YAML configuration loader for FX risk models.
"""
import yaml
from typing import Any, Dict

class ConfigLoader:
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config
