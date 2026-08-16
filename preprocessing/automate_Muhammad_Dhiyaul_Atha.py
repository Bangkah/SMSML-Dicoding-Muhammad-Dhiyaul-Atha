import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    print(f"[+] Loading raw dataset from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame):
    print("[+] Running data preprocessing pipeline...")
    
    # 1. Clean missing values
    df_clean = df.dropna().copy()
    
    # 2. Label Encoding untuk kolom kategorikal
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        
    # 3. Split Feature & Target (mengambil kolom terakhir sebagai target)
    target_col = df_clean.columns[-1]
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # 4. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"[✓] Preprocessing sukses! File CSV disimpan di: {output_dir}")

if __name__ == "__main__":
    RAW_PATH = "namadataset_raw/dataset.csv"
    OUTPUT_PATH = "Membangun_model/namadataset_preprocessing"
    
    try:
        raw_df = load_data(RAW_PATH)
        X_tr, X_te, y_tr, y_te = preprocess_data(raw_df)
        save_processed_data(X_tr, X_te, y_tr, y_te, OUTPUT_PATH)
    except Exception as e:
        print(f"[!] Terjadi kesalahan: {e}")