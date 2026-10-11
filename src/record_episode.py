
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import DQN


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "dqn_lunarlander_E08_learning_rate_0001.zip"
)
VIDEO_DIR = PROJECT_DIR / "outputs" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = VIDEO_DIR / "lunarlander_E08_episode.mp4"


def main():
    print("Chargement du modèle E08...")
    model = DQN.load(str(MODEL_PATH), device="auto")

    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    total_reward = 0.0
    episode_length = 0

    try:
        observation, _ = env.reset(seed=42)

        with imageio.get_writer(
            VIDEO_PATH,
            fps=30,
            codec="libx264",
            macro_block_size=1,
        ) as writer:
            terminated = False
            truncated = False

            while not (terminated or truncated):
                frame = env.render()
                writer.append_data(frame)

                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                action = int(np.asarray(action).item())

                observation, reward, terminated, truncated, _ = env.step(
                    action
                )

                total_reward += float(reward)
                episode_length += 1

        print("Vidéo créée :", VIDEO_PATH)
        print("Récompense totale :", round(total_reward, 3))
        print("Nombre de pas :", episode_length)
        print("Taille du fichier :", round(VIDEO_PATH.stat().st_size / 1024, 2), "Ko")

    finally:
        env.close()


if __name__ == "__main__":
    main()

