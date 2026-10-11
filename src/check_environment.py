from pathlib import Path

import gymnasium
import stable_baselines3
import torch
import fastapi
import streamlit
import imageio

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "dqn_lunarlander_E08_learning_rate_0001.zip"
)

print("Gymnasium :", gymnasium.__version__)
print("Stable-Baselines3 :", stable_baselines3.__version__)
print("PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
print("Modèle E08 présent :", MODEL_PATH.exists())

if MODEL_PATH.exists():
    print(
        "Taille du modèle :",
        round(MODEL_PATH.stat().st_size / 1024**2, 2),
        "Mo",
    )

print("FastAPI :", fastapi.__version__)
print("Streamlit :", streamlit.__version__)
print("imageio :", imageio.__version__)

