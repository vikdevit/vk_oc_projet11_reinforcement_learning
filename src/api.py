from fastapi.responses import FileResponse
from src.simulation import run_episode

from pathlib import Path

import gymnasium as gym
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from stable_baselines3 import DQN


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "dqn_lunarlander_E08_learning_rate_0001.zip"
)

app = FastAPI(
    title="LunarLander RL API",
    description="API de prédiction d'actions avec le modèle DQN E08.",
    version="1.0.0",
)

# Le modèle est chargé une seule fois au démarrage de l'API.
model = None


class ObservationRequest(BaseModel):
    observation: list[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="Observation LunarLander composée de 8 valeurs.",
    )


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Modèle introuvable : {MODEL_PATH}")

    model = DQN.load(str(MODEL_PATH), device="auto")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_PATH.name,
    }


@app.post("/predict")
def predict_action(request: ObservationRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas encore chargé.",
        )

    observation = np.asarray(request.observation, dtype=np.float32)

    if not np.all(np.isfinite(observation)):
        raise HTTPException(
            status_code=422,
            detail="L'observation doit contenir uniquement des valeurs finies.",
        )

    try:
        action, _ = model.predict(observation, deterministic=True)
        action = int(np.asarray(action).item())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {exc}",
        ) from exc

    return {"action": action}

@app.post("/simulate")
def simulate_episode(seed: int = 42, record_video: bool = False):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas encore chargé.",
        )

    try:
        return run_episode(
            model,
            seed=seed,
            record_video=record_video,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la simulation : {exc}",
        ) from exc


@app.get("/video/{seed}")
def get_episode_video(seed: int):
    video_path = (
        PROJECT_DIR
        / "outputs"
        / "videos"
        / f"lunarlander_episode_seed_{seed}.mp4"
    )

    if not video_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Vidéo introuvable pour cette seed. Lance d'abord une simulation avec vidéo.",
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=video_path.name,
    )



@app.get("/")
def root():
    return {
        "message": "LunarLander RL API opérationnelle.",
        "documentation": "/docs",
    }
