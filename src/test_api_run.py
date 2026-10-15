import requests
import gymnasium as gym


API_URL = "http://127.0.0.1:8000"


def main():
    print("Vérification de l'API...")

    response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    response.raise_for_status()

    health = response.json()

    print("API :", health["status"])
    print("Modèle chargé :", health["model_loaded"])
    print("Modèle :", health["model_name"])

    env = gym.make("LunarLander-v3")

    try:
        observation, _ = env.reset(seed=42)

        total_reward = 0.0
        episode_length = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # L'état est envoyé à l'API.
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "observation": observation.tolist()
                },
                timeout=5,
            )

            response.raise_for_status()

            result = response.json()
            action = result["action"]

            # L'action retournée par l'API
            # est appliquée à l'environnement.
            observation, reward, terminated, truncated, _ = env.step(
                action
            )

            total_reward += float(reward)
            episode_length += 1

        print()
        print("Run complet terminé.")
        print("Récompense totale :", round(total_reward, 3))
        print("Nombre d'étapes :", episode_length)
        print("Terminé :", terminated)
        print("Tronqué :", truncated)

    finally:
        env.close()


if __name__ == "__main__":
    main()
