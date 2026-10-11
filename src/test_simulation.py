
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

    print("\nTest 1 : simulation sans vidéo")
    result = run_episode(model, seed=42, record_video=False)

    print("Récompense :", result["total_reward"])
    print("Nombre d'étapes :", result["episode_length"])
    print("Répartition des actions :", result["action_counts"])

    assert result["seed"] == 42
    assert result["episode_length"] > 0
    assert sum(result["action_counts"].values()) == result["episode_length"]
    assert "video_path" not in result

    print("Test sans vidéo réussi.")

    print("\nTest 2 : simulation avec vidéo")
    result_video = run_episode(model, seed=42, record_video=True)

    print("Récompense :", result_video["total_reward"])
    print("Nombre d'étapes :", result_video["episode_length"])
    print("Chemin vidéo :", result_video["video_path"])

    video_path = Path(result_video["video_path"])

    assert video_path.is_file(), "Le fichier vidéo n'a pas été créé."
    assert video_path.stat().st_size > 0, "Le fichier vidéo est vide."
    assert result_video["seed"] == 42
    assert result_video["episode_length"] > 0

    print("Taille vidéo :", round(video_path.stat().st_size / 1024, 2), "Ko")
    print("Test avec vidéo réussi.")

    print("\nTous les tests sont réussis.")


if __name__ == "__main__":
    main()

