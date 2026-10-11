
import numpy as np
import gymnasium as gym


def run_episode(model, seed: int | None = None) -> dict:
    """Exécute un épisode LunarLander avec le modèle fourni."""

    env = gym.make("LunarLander-v3")

    try:
        observation, _ = env.reset(seed=seed)

        total_reward = 0.0
        episode_length = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # L'inférence RL reste côté backend.
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
            action_counts[action] += 1

        return {
            "seed": seed,
            "total_reward": round(total_reward, 3),
            "episode_length": episode_length,
            "terminated": terminated,
            "truncated": truncated,
            "action_counts": action_counts,
        }

    finally:
        env.close()

