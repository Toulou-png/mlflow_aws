import streamlit as st
import requests
import pandas as pd
import io

# -------------------------------------------------
# 🌐 Configuration
# -------------------------------------------------
API_URL = "http://localhost:8001"  # ⚠️ ou http://<IP_EC2>:8001 si API sur serveur distant

st.set_page_config(page_title="Aynid ML Pipeline", layout="wide")

st.title("🧠 Aynid ML Pipeline - Monitoring, Prédiction & Téléchargement")
st.markdown("Interface Streamlit complète pour interagir avec le pipeline ML FastAPI (Entraînement, Prédiction, Visualisation).")

# -------------------------------------------------
# ⚙️ ENTRAÎNEMENT DU MODÈLE
# -------------------------------------------------
st.header("⚙️ Entraîner le modèle")

n_samples = st.slider("Nombre d'échantillons à générer", 1000, 10000, 5000, 500)
train_button = st.button("🚀 Lancer l'entraînement")

if train_button:
    with st.spinner("Entraînement en cours..."):
        try:
            response = requests.post(f"{API_URL}/train", json={"n_samples": n_samples})
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Entraînement terminé avec succès !")
                st.subheader("📈 Métriques du modèle :")
                st.json(result["metrics"])

                # Sauvegarde locale des données pour téléchargement
                st.session_state["train_data"] = result.get("train_data")
                st.session_state["test_data"] = result.get("test_data")
                st.session_state["raw_data"] = result.get("raw_data")
                st.session_state["metrics"] = result.get("metrics")

            else:
                st.error(f"Erreur API : {response.text}")
        except Exception as e:
            st.error(f"❌ Impossible de contacter l'API : {e}")

# -------------------------------------------------
# 📊 VISUALISATION & TÉLÉCHARGEMENT DES DONNÉES
# -------------------------------------------------
st.header("📊 Données générées & téléchargements")

if "raw_data" in st.session_state:
    st.subheader("🧾 Données brutes générées")
    df_raw = pd.DataFrame(st.session_state["raw_data"])
    st.dataframe(df_raw.head(20), use_container_width=True)

    csv_raw = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger données brutes (CSV)", csv_raw, "raw_data.csv", "text/csv")

if "train_data" in st.session_state and "test_data" in st.session_state:
    st.subheader("📘 Données d'entraînement")
    df_train = pd.DataFrame(st.session_state["train_data"])
    st.dataframe(df_train.head(10))

    csv_train = df_train.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger jeu d'entraînement (CSV)", csv_train, "train_data.csv", "text/csv")

    st.subheader("📗 Données de test")
    df_test = pd.DataFrame(st.session_state["test_data"])
    st.dataframe(df_test.head(10))

    csv_test = df_test.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger jeu de test (CSV)", csv_test, "test_data.csv", "text/csv")

if "metrics" in st.session_state:
    st.subheader("📉 Métriques enregistrées")
    df_metrics = pd.DataFrame([st.session_state["metrics"]])
    st.dataframe(df_metrics)
    csv_metrics = df_metrics.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger métriques (CSV)", csv_metrics, "metrics.csv", "text/csv")

# -------------------------------------------------
# 🔮 PRÉDICTION D’UN UTILISATEUR
# -------------------------------------------------
st.header("🔮 Prédiction utilisateur")

st.markdown("Remplis les caractéristiques ci-dessous pour prédire si un utilisateur **abandonnera** ou **finalisera** son achat.")

col1, col2 = st.columns(2)
with col1:
    session_duration = st.number_input("⏱️ Durée de session (sec)", 10, 2000, 300)
    pages_visited = st.number_input("📄 Pages visitées", 1, 50, 8)
    cart_value = st.number_input("💰 Valeur du panier (€)", 0, 500, 75)
    time_of_day = st.slider("🕒 Heure de la journée", 0, 23, 14)

with col2:
    device_mobile = st.selectbox("📱 Utilise un mobile ?", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    user_returning = st.selectbox("🔁 Client récurrent ?", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    items_in_cart = st.number_input("🛍️ Nombre d'articles", 1, 20, 3)

if st.button("🎯 Faire la prédiction"):
    data = {
        "session_duration": session_duration,
        "pages_visited": pages_visited,
        "cart_value": cart_value,
        "time_of_day": time_of_day,
        "device_mobile": device_mobile,
        "user_returning": user_returning,
        "items_in_cart": items_in_cart
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        if response.status_code == 200:
            prediction = response.json()
            proba = prediction["probability"]
            result = prediction["prediction"]

            st.subheader("🧾 Résultat de la prédiction :")
            if result == 1:
                st.error(f"❌ L’utilisateur **risque d’abandonner** son panier ({proba*100:.1f}%)")
            else:
                st.success(f"🛒 L’utilisateur **devrait finaliser l’achat** ({(1-proba)*100:.1f}%)")
        else:
            st.error(f"Erreur API : {response.text}")
    except Exception as e:
        st.error(f"❌ Impossible de contacter l'API : {e}")
