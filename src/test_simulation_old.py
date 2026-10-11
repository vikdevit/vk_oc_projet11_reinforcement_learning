
from pathlib import Path

from stable_baselines3 import DQN

from src.simulation import run_episode


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "dqn_lunarlander_E08_learning_rate_0001.zip"
)


def main():
    print("Chargement du modèle E08...")
    model = DQN.load(str(MODEL_PATH), device="auto")
    print("Modèle chargé.")

    print("Lancement d'un épisode LunarLander...")
    result = run_episode(model, seed=42)

    print("\nRésultat de la simulation :")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

