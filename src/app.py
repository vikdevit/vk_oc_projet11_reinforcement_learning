
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"
VIDEO_PATH = "outputs/videos/lunarlander_E08_episode.mp4"

st.set_page_config(
    page_title="LunarLander - Reinforcement Learning",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 LunarLander — Reinforcement Learning")
st.markdown(
    "Explore un agent intelligent entraîné avec **Deep Q-Network (DQN)**."
)

# Vérification de l'API
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    response.raise_for_status()
    health = response.json()

    if health["model_loaded"]:
        st.success(f"API opérationnelle — Modèle : {health['model_name']}")
    else:
        st.error("L'API répond, mais le modèle n'est pas chargé.")
except requests.RequestException:
    st.error(
        "Impossible de joindre l'API FastAPI. "
        "Vérifie que le serveur est démarré."
    )
    st.stop()

st.divider()
st.subheader("Lancer une simulation")

seed = st.number_input(
    "Graine aléatoire (seed)",
    min_value=0,
    max_value=999999,
    value=42,
    step=1,
)

if st.button("🚀 Lancer l'épisode", type="primary"):
    with st.spinner("Simulation en cours..."):
        try:
            response = requests.post(
                f"{API_URL}/simulate",
                params={
                    "seed": int(seed),
                    "record_video": True,
                },
                timeout=180,
            )
            response.raise_for_status()
            result = response.json()
            st.session_state["simulation_result"] = result
        except requests.RequestException as exc:
            st.error(f"Erreur pendant la simulation : {exc}")

result = st.session_state.get("simulation_result")

if result:
    st.divider()
    st.subheader("Résultats de la simulation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Récompense totale", result["total_reward"])
    col2.metric("Nombre d'étapes", result["episode_length"])
    col3.metric("Seed", result["seed"])

    
    st.write("**Actions choisies par l'agent**")

    noms_actions = {
        0: "Aucune action",
        1: "Moteur gauche",
        2: "Moteur principal",
        3: "Moteur droit",
    }

    action_counts = result["action_counts"]
    action_counts_fr = {
        noms_actions.get(int(action), str(action)): nombre
        for action, nombre in action_counts.items()
    }

    st.bar_chart(action_counts_fr)


#    st.write("**Actions choisies par l'agent**")
#    st.bar_chart(result["action_counts"])

#st.divider()
#st.subheader("Vidéo de démonstration")
#
#st.caption(
#    "Vidéo préenregistrée avec le modèle E08 et la seed 42."
#)
#
#try:
#    with open(VIDEO_PATH, "rb") as video_file:
#        st.video(video_file.read())
#except FileNotFoundError:
#    st.warning(
#        "Vidéo introuvable. Vérifie le fichier "
#        "outputs/videos/lunarlander_E08_episode.mp4."
#    )


# Courbes d'évaluation du modèle E08
import pandas as pd
from pathlib import Path

st.divider()
st.subheader("📈 Évaluation du modèle DQN E08")

eval_file = Path("outputs/metrics/dqn_lunarlander_E08_eval_100.csv")

if eval_file.exists():
    df_eval = pd.read_csv(eval_file)

    # Récompenses obtenues sur les 100 épisodes d'évaluation
    st.write("**Récompense par épisode d'évaluation**")
    st.line_chart(
        df_eval.set_index("seed")[["reward"]],
        x_label="Seed",
        y_label="Récompense totale",
    )

    # Nombre d'étapes par épisode
    st.write("**Nombre d'étapes par épisode d'évaluation**")
    st.line_chart(
        df_eval.set_index("seed")[["episode_length"]],
        x_label="Seed",
        y_label="Nombre d'étapes",
    )

    st.caption(
        "Résultats de l'évaluation du modèle E08 sur 100 épisodes."
    )
else:
    st.warning("Le fichier d'évaluation E08 est introuvable.")



st.divider()
st.subheader("Vidéo de l'épisode simulé")

if result:
    try:
        video_response = requests.get(
            f"{API_URL}/video/{result['seed']}",
            timeout=30,
        )
        video_response.raise_for_status()

        st.video(video_response.content)

        st.caption(
            f"Vidéo générée par le backend pour la seed {result['seed']}."
        )

    except requests.RequestException as exc:
        st.error(f"Impossible de récupérer la vidéo : {exc}")
else:
    st.info("Lance un épisode pour afficher sa vidéo.")

st.caption("Projet OpenClassrooms — Reinforcement Learning")




