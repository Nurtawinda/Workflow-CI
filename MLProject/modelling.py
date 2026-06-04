import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import warnings
import shutil

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    # 1. Jalur Pelacakan URI
    if os.environ.get("GITHUB_ACTIONS") == "true":
        mlflow.set_tracking_uri("file:./mlruns")
    else:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "Nurtawinda"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "e6481f90867f23522774600ba572268d89071390"
        mlflow.set_tracking_uri("https://dagshub.com")
    
    mlflow.set_experiment("Submission SML winda")
    
    n_estimators = 505
    max_depth = 37
    
    # 2. Deteksi Otomatis Jalur Dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "credit_risk_dataset_preprocessing.csv")
    
    if not os.path.exists(file_path):
        file_path = "MLProject/credit_risk_dataset_preprocessing.csv"
        
    print(f"-> Membaca dataset dari jalur: {file_path}")
    data = pd.read_csv(file_path)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("loan_status", axis=1),
        data["loan_status"],
        random_state=42,
        test_size=0.2
    )
    input_example = X_train[0:5]
    
    # 3. Proses Training Model
    with mlflow.start_run(run_name=f"MLProject_est_{n_estimators}_depth_{max_depth}", nested=True):
        mlflow.autolog(disable=True)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        # Simpan ke log artefak internal
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example)
        
        # Log Metrik Akurasi
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        # 4. Ambil folder model fisik lokal ke dalam folder MLProject/model
        local_model_path = os.path.join(base_dir, "model")
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
            
        mlflow.sklearn.save_model(sk_model=model, path=local_model_path)
        print(f"-> Sukses! Model fisik lokal disimpan di: {local_model_path}")

    print(f"-> Selesai Sempurna! Akurasi Model Final: {accuracy:.4f}")
