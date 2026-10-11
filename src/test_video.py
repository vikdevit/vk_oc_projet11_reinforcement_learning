
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio


PROJECT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_DIR / "outputs" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = VIDEO_DIR / "test_render.mp4"


def main():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    try:
        observation, _ = env.reset(seed=42)

        with imageio.get_writer(
            VIDEO_PATH,
            fps=30,
            codec="libx264",
        ) as writer:
            for _ in range(90):
                frame = env.render()
                writer.append_data(frame)

                # Action neutre : test du rendu et de l'encodage.
                observation, reward, terminated, truncated, _ = env.step(0)

                if terminated or truncated:
                    observation, _ = env.reset()

        print("Vidéo créée :", VIDEO_PATH)
        print("Taille :", round(VIDEO_PATH.stat().st_size / 1024, 2), "Ko")

    finally:
        env.close()


if __name__ == "__main__":
    main()

