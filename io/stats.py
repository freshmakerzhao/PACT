import os
import pickle
from typing import Dict


def save_stats(stats: Dict[str, object], ckpt_dir: str) -> str:
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    return stats_path


def load_stats(ckpt_dir: str) -> Dict[str, object]:
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        return pickle.load(f)

