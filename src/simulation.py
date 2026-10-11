
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_DIR / "outputs" / "videos"


def run_episode(
    model,
    seed: int | None = None,
    record_video: bool = False,
) -> dict:
    """Exécute un épisode LunarLander et peut enregistrer sa vidéo."""

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array" if record_video else None,
    )

    video_path = (
        VIDEO_DIR / f"lunarlander_episode_seed_{seed}.mp4"
        if record_video
        else None
    )

    total_reward = 0.0
    episode_length = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    try:
        observation, _ = env.reset(seed=seed)

        terminated = False
        truncated = False

        if record_video:
            with imageio.get_writer(
                video_path,
                fps=30,
                codec="libx264",
                macro_block_size=1,
            ) as writer:
                while not (terminated or truncated):
                    writer.append_data(env.render())

                    action, _ = model.predict(
                        observation,
                        deterministic=True,
                    )
                    action = int(np.asarray(action).item())

                    observation, reward, terminated, truncated, _ = (
                        env.step(action)
                    )

                    total_reward += float(reward)
                    episode_length += 1
                    action_counts[action] += 1
        else:
            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )
                action = int(np.asarray(action).item())

                observation, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += float(reward)
                episode_length += 1
                action_counts[action] += 1

        result = {
            "seed": seed,
            "total_reward": round(total_reward, 3),
            "episode_length": episode_length,
            "terminated": terminated,
            "truncated": truncated,
            "action_counts": action_counts,
        }

        if video_path is not None:
            result["video_path"] = str(video_path)

        return result

    finally:
        env.close()

