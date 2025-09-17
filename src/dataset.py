import pandas as pd, numpy as np, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

def split_dataframe(df, split_col='split', train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    if split_col in df.columns and df[split_col].notna().any():
        return df[df[split_col]=='train'].copy(), df[df[split_col]=='val'].copy(), df[df[split_col]=='test'].copy()
    assert abs(train_size+val_size+test_size-1.0) < 1e-9
    train_full, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    train_df, val_df = train_test_split(train_full, test_size=val_size/(train_size+val_size), random_state=random_state, shuffle=True)
    return train_df, val_df, test_df

def build_matrices(df, feature_cols, target_cols):
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_cols].to_numpy(dtype=np.float32)
    return X, y

def fit_scale_save(X_train, scaler_path):
    sc = StandardScaler().fit(X_train)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    dump(sc, scaler_path)
    return sc

def load_scaler(path):
    return load(path)
