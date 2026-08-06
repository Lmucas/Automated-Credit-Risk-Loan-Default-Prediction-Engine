import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def load_raw_data(data_path: str = "data/raw/credit_risk_dataset.csv") -> pd.DataFrame:
    """Loads raw credit risk dataset from CSV file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at path: {data_path}")
    df = pd.read_csv(data_path)
    print(f" Raw dataset loaded successfully. Shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes credit risk and financial stability ratios."""
    df = df.copy()
    
    # 1. Debt-to-Income (DTI) ratio proxy (Loan amount / Annual Income)
    df["dti_ratio"] = df["loan_amnt"] / (df["person_income"] + 1e-5)
    
    # 2. Income-to-Age ratio
    df["income_to_age"] = df["person_income"] / (df["person_age"] + 1e-5)
    
    # 3. Loan-to-Employment-Length ratio
    df["loan_to_emp_length"] = df["loan_amnt"] / (df["person_emp_length"] + 1)
    
    print(" Feature engineering complete. Engineered 3 new financial ratios.")
    return df


def preprocess_data(df: pd.DataFrame, target_col: str = "loan_status"):
    """
    Imputes missing values, encodes categorical features, and prepares 
    feature matrix X and target array y.
    """
    df = df.copy()
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Separate numerical and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Median imputation for numerical features
    num_imputer = SimpleImputer(strategy="median")
    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    
    # One-Hot Encoding for categorical features
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    print(f" Preprocessing complete. Final feature matrix shape: {X_encoded.shape}")
    return X_encoded, y


def split_and_balance_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42):
    """
    Splits data into train and test sets, then applies SMOTE on the training set ONLY
    to prevent data leakage into the evaluation set.
    """
    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f" Pre-SMOTE Train Set: {X_train.shape[0]} samples (Defaults: {y_train.sum()})")
    
    # Apply SMOTE only on training data
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f" Post-SMOTE Train Set: {X_train_resampled.shape[0]} samples (Defaults: {y_train_resampled.sum()})")
    print(f" Test Set (Unchanged): {X_test.shape[0]} samples")
    
    return X_train_resampled, X_test, y_train_resampled, y_test


def run_data_pipeline(data_path: str = "data/raw/credit_risk_dataset.csv"):
    """Full execution pipeline wrapper."""
    df = load_raw_data(data_path)
    df_engineered = engineer_features(df)
    X, y = preprocess_data(df_engineered)
    X_train_res, X_test, y_train_res, y_test = split_and_balance_data(X, y)
    
    return X_train_res, X_test, y_train_res, y_test


if __name__ == "__main__":
    # Test run the pipeline locally
    X_train, X_test, y_train, y_test = run_data_pipeline()
