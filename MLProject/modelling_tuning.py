import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import warnings
import argparse
import shutil

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    parser = argparse.ArgumentParser()
    parser.add_argument("n_estimators", type=int, nargs="?", default=505)
    parser.add_argument("max_depth", type=int, nargs="?", default=37)
    parser.add_argument("dataset", type=str, nargs="?", default="credit_risk_dataset_preprocessing.csv")
    args = parser.parse_args()

    if os.environ.get("GITHUB_ACTIONS") == "true":
        mlflow.set_tracking_uri("file:./mlruns")
    else:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Nurtawinda"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "e6481f90867f23522774600ba572268d89071390"
        DAGSHUB_MLFLOW_URI = "https://dagshub.com"
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    
    mlflow.set_experiment("Submission SML winda")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, args.dataset)
    
    if not os.path.exists(file_path):
        file_path = os.path.join("MLProject", "credit_risk_dataset_preprocessing.csv")

    print(f"-> Mengakses dataset dari jalur: {file_path}")
    data = pd.read_csv(file_path)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("loan_status", axis=1),
        data["loan_status"],
        random_state=42,
        test_size=0.2
    )
    
    input_example = X_train[0:5]
    
    with mlflow.start_run(run_name=f"MLProject_est_{args.n_estimators}_depth_{args.max_depth}"):
        mlflow.autolog(disable=True)

        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        # Log Model ke Artefak MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )
        
        # Log Metrik Akurasi
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        local_model_path = os.path.join(base_dir, "model")
        
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        
        mlflow.sklearn.save_model(sk_model=model, path=local_model_path)
        print(f"-> Sukses! Model fisik lokal berhasil disimpan di: {local_model_path}")

    print(f"Proses Run MLProject Selesai Sempurna! Akurasi Data Uji: {accuracy:.4f}")
