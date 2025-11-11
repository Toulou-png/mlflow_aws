En tant qu'assistant IA, je peux générer un fichier **README.md** complet pour votre projet **`mlflow_aws`** basé sur les fichiers Python, JSON et Bash que vous avez fournis.

## 🧠 Projet : Pipeline ML Aynid sur AWS avec MLflow et Observabilité

Ce projet implémente un pipeline d'apprentissage automatique (Machine Learning - ML) pour la **prédiction d'abandon de panier** pour la société fictive Aynid. L'infrastructure est déployée sur **AWS** et utilise **MLflow** pour le suivi des expériences et **Prometheus/Grafana** pour l'observabilité. Une API **FastAPI** permet l'entraînement et la prédiction, et une interface **Streamlit** offre un tableau de bord convivial.

-----

### 📂 Structure du Projet

```
mlflow_aws/
├── api_pipeline.py            # API FastAPI pour le pipeline ML (Entraînement & Prédiction)
├── aynid_pipeline.py          # Logique du pipeline ML (Préparation des données, Entraînement, Logging)
├── mlflow_aws.py              # Configuration AWS/MLflow/Postgres (similaire à aynid_pipeline.py, contient aussi des plots)
├── mlflow_cg.sh               # Script Bash pour l'export des variables d'environnement MLflow
├── aynid_ml_dashboard.json    # Tableau de bord Grafana pour le monitoring Prometheus
├── streamlit_app.py           # Interface utilisateur Streamlit pour interagir avec l'API
├── requirements.txt           # Dépendances Python
└── test_customer_data.csv     # Exemple de données client (générées mais incluses pour référence)
```

-----

### 🚀 Démarrage Rapide

#### prerequisites

  * Compte AWS avec accès aux services **S3** et **RDS (PostgreSQL)**.
  * Instance **EC2** pour héberger le serveur MLflow et le serveur Prometheus/Grafana.
  * **Docker** et **Docker Compose** (si utilisation d'une approche conteneurisée).
  * **Python 3.10+** avec les dépendances listées dans `requirements.txt`.

#### 🛠️ Configuration des Environnements

Le script `mlflow_cg.sh` contient les configurations principales. **Mettez à jour** les variables d'environnement suivantes avec vos propres valeurs :

```bash
# mlflow_cg.sh (à adapter)
export MLFLOW_S3_ENDPOINT_URL=https://s3.eu-west-3.amazonaws.com # Région S3
export AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID # Clé d'accès AWS
export AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY # Clé secrète AWS
export BACKEND_URI=postgresql://postgres:Pida2025@mlflow-postgre-db.c09u0wy6mlax.us-east-1.rds.amazonaws.com:5432/mlflow # Connexion PostgreSQL
export ARTIFACT_URI=s3://mlflow-artefacts-aynid # Nom du bucket S3
export MLFLOW_TRACKING_URI="http://3.85.105.94:5000" # IP/Port de l'instance MLflow
```

Sourcez le script : `source mlflow_cg.sh`

#### ⚙️ Lancement des Services

1.  **Lancer le serveur MLflow** (souvent sur l'instance EC2, port `5000`) :

    ```bash
    mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $BACKEND_URI --default-artifact-root $ARTIFACT_URI
    ```

2.  **Lancer l'API FastAPI** (pour l'entraînement et la prédiction, port `8001`) :

    ```bash
    python -m uvicorn api_pipeline:app --host 0.0.0.0 --port 8001 --reload
    ```

    *Note: L'API lance également l'**Exporter Prometheus** sur le port `8000` via `aynid_pipeline.py`.*

3.  **Configurer Prometheus & Grafana** :

      * Le fichier `prometheus.yaml` doit pointer vers l'**adresse IP publique de l'instance EC2** hébergeant l'API sur le port `8000` (voir `scrape_configs`).
      * Importez le dashboard Grafana `aynid_ml_dashboard.json` et configurez une source de données Prometheus pointant vers votre serveur Prometheus.

4.  **Lancer le tableau de bord Streamlit** (depuis votre machine locale, si l'API est accessible) :

    ```bash
    streamlit run streamlit_app.py
    ```

      * **Attention** : Mettez à jour `API_URL` dans `streamlit_app.py` avec l'IP publique de votre instance EC2 si l'API n'est pas locale.

-----

### 💻 Utilisation des Composants Clés

#### 📊 Pipeline ML (`aynid_pipeline.py` / `mlflow_aws.py`)

Ce module gère le cycle de vie ML complet :

  * **Connexion aux Services :** S3 (artefacts), PostgreSQL (métriques personnalisées), MLflow (tracking).
  * **Préparation des données :** Génère des données synthétiques pour la prédiction d'abandon de panier.
  * **Entraînement du modèle :** Utilise un **RandomForestClassifier**.
  * **Tracking MLflow :** Log des paramètres, métriques et du modèle.
  * **Monitoring Prometheus :** Met à jour les métriques exposées sur le port `8000` (Accuracy, F1-Score, etc.).
  * **Persistance des Métriques :** Sauvegarde des métriques clés dans une table **PostgreSQL** (`custom_metrics`).

#### 🌐 API FastAPI (`api_pipeline.py`)

L'API expose trois endpoints principaux :

| Endpoint | Méthode | Description | Payload (Exemple) |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | Message de bienvenue. | - |
| `/train` | `POST` | Lance la préparation des données et l'entraînement du modèle. Sauvegarde le modèle localement (`model_latest.pkl`). | `{"n_samples": 5000}` |
| `/predict` | `POST` | Effectue une prédiction en utilisant `model_latest.pkl`. | `{"session_duration": 350.0, "pages_visited": 8, "cart_value": 75.0, "time_of_day": 14, "device_mobile": 1, "user_returning": 1, "items_in_cart": 3}` |
| `/metrics`| `GET` | Affiche un message de statut pour le monitoring Prometheus. | - |

#### 📈 Dashboard Streamlit (`streamlit_app.py`)

Une interface simple pour :

1.  **Lancer un nouvel entraînement** via l'endpoint `/train` de l'API.
2.  **Visualiser** les métriques et des extraits des jeux de données générés.
3.  **Tester la prédiction** pour un utilisateur donné via l'endpoint `/predict`.

-----

### ☁️ Configuration AWS

Le pipeline nécessite :

1.  Un **bucket S3** (`mlflow-artefacts-aynid`) pour stocker les artefacts MLflow (modèles, métriques CSV, plots).
2.  Une base de données **PostgreSQL RDS** (ou compatible) pour le *backend* de suivi MLflow et pour stocker les métriques personnalisées du pipeline (`custom_metrics`).

Les informations de connexion à ces services sont gérées via les variables d'environnement dans `mlflow_cg.sh` et utilisées par `aynid_pipeline.py` pour configurer MLflow.

-----

### 📈 Observabilité

Le monitoring des performances du modèle est crucial :

  * **Prometheus Exporter (via `aynid_pipeline.py`):** Expose les métriques temps réel du dernier entraînement (`model_accuracy`, `model_f1_score`, etc.) sur le port `8000`.
  * **Prometheus (configuré via `prometheus.yaml`):** Scrape les métriques de l'API ML sur le port `8000` de l'instance EC2.
  * **Grafana (`aynid_ml_dashboard.json`):** Affiche les métriques collectées par Prometheus pour le suivi de la santé et de la performance du modèle.