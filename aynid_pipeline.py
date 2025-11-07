import os
import boto3
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tempfile
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from botocore.exceptions import ClientError
import psycopg2

# --- 🧩 OpenTelemetry / Prometheus ---
import logging
from prometheus_client import start_http_server, Gauge
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import Resource


IP_ADDRESS = "3.85.105.94"  # Adresse IP publique de votre instance EC2

# ===========================================================
# 🌩️ CONFIGURATION AWS + MLFLOW + POSTGRES
# ===========================================================
def setup_aws_mlflow_postgres(
    region="us-east-1",
    bucket_name="mlflow-artefacts-aynid",
    mlflow_server_uri=f"http://{IP_ADDRESS}:5000",
    postgres_config=None
):
    try:
        session = boto3.session.Session(region_name=region)
        s3_client = session.client("s3")

        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket S3 déjà présent : {bucket_name}")
        except ClientError:
            print(f"🪣 Création du bucket S3 : {bucket_name} ...")
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )

        os.environ["MLFLOW_TRACKING_URI"] = mlflow_server_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"https://s3.{region}.amazonaws.com"
        os.environ["MLFLOW_S3_BUCKET"] = bucket_name
        os.environ["AWS_DEFAULT_REGION"] = region

        mlflow.set_tracking_uri(mlflow_server_uri)

        print(f"\n🔧 MLflow configuré : {mlflow_server_uri}")
        print(f"S3 bucket → {bucket_name}")

        if postgres_config:
            try:
                conn = psycopg2.connect(**postgres_config)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS custom_metrics (
                        id SERIAL PRIMARY KEY,
                        run_id VARCHAR(64),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accuracy FLOAT,
                        precision FLOAT,
                        recall FLOAT,
                        f1_score FLOAT,
                        auc_score FLOAT,
                        s3_path TEXT
                    );
                """)
                conn.commit()
                cursor.close()
                conn.close()
                print("✅ Connexion PostgreSQL OK et table prête.")
            except Exception as e:
                print(f"⚠️ PostgreSQL non accessible : {e}")

    except Exception as e:
        print(f"❌ Erreur config AWS/MLflow/Postgres : {e}")


# ===========================================================
# 📈 CONFIGURATION OBSERVABILITÉ (Prometheus / OTel)
# ===========================================================
def setup_prometheus_exporter(port=8000):
    resource = Resource(attributes={"service.name": "ml-pipeline-aynid"})
    reader = PrometheusMetricReader()
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter("ml_pipeline_meter")

    print(f"📡 Exporter Prometheus lancé sur port {port}")
    start_http_server(port)
    return meter


# ===========================================================
# 🧠 DÉCLARATION GLOBALE DES GAUGES PROMETHEUS (✅ Fix duplication)
# ===========================================================
model_accuracy_gauge = Gauge('model_accuracy', 'Accuracy du modèle')
model_f1_gauge = Gauge('model_f1_score', 'F1-score du modèle')
model_precision_gauge = Gauge('model_precision', 'Precision du modèle')
model_recall_gauge = Gauge('model_recall', 'Recall du modèle')
model_auc_gauge = Gauge('model_auc', 'AUC du modèle')


# ===========================================================
# ⚙️ PIPELINE MLFLOW + PROMETHEUS + POSTGRES
# ===========================================================
class AynidCartAbandonmentAWS:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm")
        self.experiment_name = "Aynid_Abandon_Panier"
        self.registry_name = "RF_Abandonment_Predictor"

        self.postgres_config = {
            "host": "mlflow-postgre-db.c09u0wy6mlax.us-east-1.rds.amazonaws.com",
            "port": 5432,
            "database": "mlflow",
            "user": "postgres",
            "password": "Pida2025"
        }

        setup_aws_mlflow_postgres(postgres_config=self.postgres_config)
        mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient()

        self.s3 = boto3.client("s3")
        self.bucket = os.getenv("MLFLOW_S3_BUCKET")

        # Démarre Prometheus Exporter
        self.meter = setup_prometheus_exporter()

        print(f"✅ MLflow connecté à : {mlflow.get_tracking_uri()}")
        print(f"🪣 Artefacts stockés sur : {self.bucket}")

    def _upload_file_to_s3(self, file_path, s3_key):
        try:
            self.s3.upload_file(file_path, self.bucket, s3_key)
            print(f"✅ Fichier envoyé sur S3 : {s3_key}")
        except Exception as e:
            print(f"❌ Erreur upload S3 ({s3_key}): {e}")

    def prepare_data(self, n_samples=5000):
        np.random.seed(42)
        data = {
            'session_duration': np.random.exponential(300, n_samples),
            'pages_visited': np.random.poisson(8, n_samples),
            'cart_value': np.random.normal(75, 25, n_samples),
            'time_of_day': np.random.randint(0, 24, n_samples),
            'device_mobile': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'user_returning': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'items_in_cart': np.random.randint(1, 10, n_samples)
        }

        abandonment_proba = (
            0.3 + 0.4 * (data['session_duration'] < 180) +
            0.2 * (data['cart_value'] > 100) -
            0.1 * data['user_returning'] +
            0.15 * (data['time_of_day'] > 18)
        )
        data['cart_abandoned'] = np.random.binomial(1, np.clip(abandonment_proba, 0, 1))
        df = pd.DataFrame(data)

        # Save data and upload
        raw_path = f"data/raw/raw_data_{self.timestamp}.csv"
        df.to_csv("raw_data.csv", index=False)
        self._upload_file_to_s3("raw_data.csv", raw_path)

        df['cart_value'] = np.clip(df['cart_value'], 0, 300)
        processed_path = f"data/processed/processed_data_{self.timestamp}.csv"
        df.to_csv("processed_data.csv", index=False)
        self._upload_file_to_s3("processed_data.csv", processed_path)
        return df

    def train_model(self, df):
        X = df.drop('cart_abandoned', axis=1)
        y = df['cart_abandoned']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        params = {"n_estimators": 300, "max_depth": 12, "random_state": 42}
        model = RandomForestClassifier(**params)

        Mlflow_tracking_URI = f"http://{IP_ADDRESS}:5000"
        mlflow.set_tracking_uri(Mlflow_tracking_URI)

        with mlflow.start_run(run_name=f"RF_{self.timestamp}") as run:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            metrics_values = {
                "auc_score": roc_auc_score(y_test, y_pred_proba),
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }

            mlflow.log_params(params)
            mlflow.log_metrics(metrics_values)

            # 🔧 Met à jour les gauges déjà créées (pas de recréation)
            model_accuracy_gauge.set(metrics_values["accuracy"])
            model_f1_gauge.set(metrics_values["f1_score"])
            model_precision_gauge.set(metrics_values["precision"])
            model_recall_gauge.set(metrics_values["recall"])
            model_auc_gauge.set(metrics_values["auc_score"])

            # PostgreSQL
            self._push_metrics_postgres(run.info.run_id, metrics_values, f"metrics/metrics_{self.timestamp}.csv")

            # Modèle
            mlflow.sklearn.log_model(model, "model", registered_model_name=self.registry_name)

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_model:
                joblib.dump(model, tmp_model.name)
                self._upload_file_to_s3(tmp_model.name, f"models/model_{self.timestamp}.pkl")

            self.create_visuals(y_test, y_pred, y_pred_proba)

        return model, metrics_values

    def _push_metrics_postgres(self, run_id, metrics, s3_path):
        try:
            conn = psycopg2.connect(**self.postgres_config)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO custom_metrics (run_id, accuracy, precision, recall, f1_score, auc_score, s3_path)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
            """, (
                run_id,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["auc_score"],
                f"s3://{self.bucket}/{s3_path}"
            ))
            conn.commit()
            cursor.close()
            conn.close()
            print(f"✅ Metrics insérées dans PostgreSQL ({run_id})")
        except Exception as e:
            print(f"❌ Erreur PostgreSQL : {e}")

    def create_visuals(self, y_test, y_pred, y_pred_proba):
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
        self._upload_file_to_s3("confusion_matrix.png", f"runs/{self.timestamp}/confusion_matrix.png")


# ===========================================================
# 🚀 EXECUTION LOCALE (pour test direct)
# ===========================================================
if __name__ == "__main__":
    pipeline = AynidCartAbandonmentAWS()
    df = pipeline.prepare_data()
    model, metrics = pipeline.train_model(df)
