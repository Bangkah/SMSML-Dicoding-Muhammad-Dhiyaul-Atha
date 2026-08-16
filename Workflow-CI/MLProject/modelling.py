import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("Basic_Model_Muhammad_Dhiyaul_Atha")

def train_basic_model():
    DATA_DIR = "Membangun_model/belajar_preprocessing"
    
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()
    
    mlflow.sklearn.autolog(log_models=True, disable=False)
    
    with mlflow.start_run(run_name="RandomForest_Basic"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"[✓] Basic Model Trained. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_basic_model()