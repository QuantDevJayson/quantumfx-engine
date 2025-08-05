"""
Reporting utilities for FX risk models.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

class FXReport:
    @staticmethod
    def plot_fx_path(path: np.ndarray, title: str = "FX Rate Path"):
        plt.figure(figsize=(10, 4))
        plt.plot(path)
        plt.title(title)
        plt.xlabel("Time Step")
        plt.ylabel("FX Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def summary_stats(path: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(path)),
            "std": float(np.std(path)),
            "min": float(np.min(path)),
            "max": float(np.max(path)),
        }
