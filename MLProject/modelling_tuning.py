import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import warnings
import sys
import shutil

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    if os.environ.get("GITHUB_ACTIONS") == "true":
        mlflow.set_tracking_uri("file:./mlruns")
    else:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Nurtawinda"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "e6481f90867f23522774600ba572268d89071390"
        DAGSHUB_MLFLOW_URI = "https://dagshub.com"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    
    mlflow.set_experiment("Submission SML winda")
    try:
        n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    except (ValueError, IndexError):
        n_estimators = 505

    try:
        max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    except (ValueError, IndexError):
        max_depth = 37

    try:
        file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_dataset_preprocessing.csv")
    except Exception:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_dataset_preprocessing.csv")

    # Membaca Dataset
    data = pd.read_csv(file_path)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("loan_status", axis=1),
        data["loan_status"],
        random_state=42,
        test_size=0.2
    )
    
    input_example = X_train[0:5]
    
    with mlflow.start_run(run_name=f"MLProject_est_{n_estimators}_depth_{max_depth}"):
        mlflow.autolog(disable=True)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        
        # Log Metrik Akurasi
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        
        mlflow.sklearn.save_model(sk_model=model, path=local_model_path)
        print(f"-> Sukses! Model lokal berhasil disimpan di: {local_model_path}")

    print(f"Proses Run MLProject Selesai Sempurna! Akurasi Data Uji: {accuracy:.4f}")
