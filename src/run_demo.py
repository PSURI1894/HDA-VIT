# Quick orchestration demo for the advanced repo using synthetic data
import os
from src.synth_demo import generate_sample_dataset
from src.train_seg import train_seg
from src.train_cls import train_cls

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "experiments", "models")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Generating synthetic dataset (this may take some minutes)...")
    generate_sample_dataset(DATA_DIR, n_cases=6, vol_size=128, seed=42)
    print("Prepare a simple train_list.json pointing to produced files before training segmentation.")
    # to run training one can prepare experiments/train_list.json and then call train_seg
