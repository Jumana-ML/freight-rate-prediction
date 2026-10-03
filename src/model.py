import numpy as np
import pandas as pd

class PureNumPyRidge:
    """
    Custom Ridge Regression implementation using closed-form matrix solution (A @ theta = b).
    Guarantees cross-platform stability without C/DLL library dependencies.
    """
    def __init__(self, alpha=10.0):
        self.alpha = alpha
        self.mean = None
        self.std = None
        self.weights = None

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        
        # Standardize numerical features
        self.mean = np.mean(X_arr, axis=0)
        self.std = np.std(X_arr, axis=0)
        self.std[self.std == 0] = 1.0  # Prevent division by zero
        
        X_scaled = (X_arr - self.mean) / self.std
        X_b = np.c_[np.ones(len(X_scaled)), X_scaled]  # Add bias term
        
        n_features = X_b.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0.0  # Do not penalize bias intercept
        
        A = X_b.T @ X_b + self.alpha * I
        b = X_b.T @ y_arr
        self.weights = np.linalg.solve(A, b)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        X_scaled = (X_arr - self.mean) / self.std
        X_b = np.c_[np.ones(len(X_scaled)), X_scaled]
        return X_b @ self.weights

def pure_kfold_split(n_samples, n_splits=5, seed=42):
    """
    Generates reproducible K-Fold indices array split.
    """
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    return np.array_split(indices, n_splits)

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

def compute_mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

def train_and_predict(train_df, val_df, dec_df, num_features, cat_features, target_col='posted_rate'):
    """
    Trains a 5-Fold Cross-Validated Log-Ridge Regression model with out-of-fold Target Encoding.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    dec_df = dec_df.copy()

    n_train = len(train_df)
    global_target_mean = train_df[target_col].mean()

    # Pre-calculate full target maps for validation and December test sets
    full_maps = {}
    for col in cat_features:
        full_maps[col] = train_df.groupby(col)[target_col].mean().to_dict()

    val_encoded_cols = [val_df[col].values for col in num_features]
    dec_encoded_cols = [dec_df[col].values for col in num_features]

    for col in cat_features:
        val_encoded_cols.append(val_df[col].map(full_maps[col]).fillna(global_target_mean).values)
        dec_encoded_cols.append(dec_df[col].map(full_maps[col]).fillna(global_target_mean).values)

    X_val_mat = np.column_stack(val_encoded_cols)
    X_dec_mat = np.column_stack(dec_encoded_cols)

    folds = pure_kfold_split(n_train, n_splits=5, seed=42)
    
    oof_preds = np.zeros(n_train)
    val_preds = np.zeros(len(val_df))
    dec_preds = np.zeros(len(dec_df))

    print("Training Log-Transformed Ridge Model with Out-Of-Fold Target Encoding...")

    for fold_idx, test_indices in enumerate(folds):
        train_mask = np.ones(n_train, dtype=bool)
        train_mask[test_indices] = False
        
        train_fold = train_df.iloc[train_mask]
        valid_fold = train_df.iloc[test_indices]

        # Compute Target Encoding strictly within training fold to prevent data leakage
        fold_maps = {}
        for col in cat_features:
            fold_maps[col] = train_fold.groupby(col)[target_col].mean().to_dict()

        tr_cols = [train_fold[col].values for col in num_features]
        for col in cat_features:
            tr_cols.append(train_fold[col].map(fold_maps[col]).fillna(global_target_mean).values)
        X_tr = np.column_stack(tr_cols)
        
        # Apply natural log transformation on target rate
        y_tr_log = np.log(train_fold[target_col].values)

        va_cols = [valid_fold[col].values for col in num_features]
        for col in cat_features:
            va_cols.append(valid_fold[col].map(fold_maps[col]).fillna(global_target_mean).values)
        X_va = np.column_stack(va_cols)
        y_va = valid_fold[target_col].values

        model = PureNumPyRidge(alpha=10.0)
        model.fit(X_tr, y_tr_log)

        # Inverse log transform via np.exp to restore predictions to dollar amounts
        val_fold_pred = np.exp(model.predict(X_va))
        oof_preds[test_indices] = val_fold_pred

        val_preds += np.exp(model.predict(X_val_mat)) / 5.0
        dec_preds += np.exp(model.predict(X_dec_mat)) / 5.0

        fold_rmse = compute_rmse(y_va, val_fold_pred)
        print(f"  --> Fold {fold_idx + 1} RMSE: ${fold_rmse:.4f}")

    # Clip predictions to minimum observed historical rate
    min_rate_train = float(train_df[target_col].min())
    val_preds = np.clip(val_preds, a_min=min_rate_train, a_max=None)
    dec_preds = np.clip(dec_preds, a_min=min_rate_train, a_max=None)

    overall_rmse = compute_rmse(train_df[target_col].values, oof_preds)
    overall_mae = compute_mae(train_df[target_col].values, oof_preds)

    print(f"\n Cross-Validation Performance Metrics:")
    print(f"   - Overall RMSE: ${overall_rmse:.4f}")
    print(f"   - Overall MAE:  ${overall_mae:.4f}")

    return val_preds, dec_preds