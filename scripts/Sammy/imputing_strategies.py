
import polars as pl
from libs.io_lib.paths import LAGS_FEATURES_TRAINING, LAGS_FEATURES_VALIDATION, SAMPLE_LAGS_FEATURES_TRAINING
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import numpy as np

feature_names = [f"feature_{i:02d}" for i in range(79)] + [f"feature_{i:02d}_lag_1" for i in range(79)] + [f"responder_{idx}_lag_1" for idx in range(9)]
label_name = 'responder_6'
weight_name = 'weight'

# Load data
df = pl.scan_parquet(SAMPLE_LAGS_FEATURES_TRAINING).collect().to_pandas()
valid = pl.scan_parquet(LAGS_FEATURES_VALIDATION).collect().to_pandas()



### Here is when u want to select only one symbol 
'''
sym = 1
df_sym = df[df['symbol_id'] == sym]
valid_sym = valid[valid['symbol_id'] == sym]
'''


### here we check across all symbols

df_sym = df 
valid_sym = valid

X_train = df_sym[feature_names].to_numpy()
y_train = df_sym[label_name].to_numpy().ravel()
w_train = df_sym["weight"].to_numpy().ravel()
X_valid = valid_sym[feature_names].to_numpy()
y_valid = valid_sym[label_name].to_numpy().ravel()
w_valid = valid_sym["weight"].to_numpy().ravel()


df_zero = df_sym.copy()
df_median = df_sym.copy()
df_mean = df_sym.copy()

# Apply different filling strategies
df_zero.loc[:, feature_names] = df_zero.groupby(['date_id', 'symbol_id'])[feature_names].ffill().fillna(0)
df_median.loc[:, feature_names] = df_median.groupby(['date_id', 'symbol_id'])[feature_names].ffill().fillna(df_median[feature_names].median())
df_mean.loc[:, feature_names] = df_mean.groupby(['date_id', 'symbol_id'])[feature_names].ffill().fillna(df_mean[feature_names].mean())


print("\nSample means comparison:")
print(f"Zero strategy mean: {df_zero[feature_names].mean().mean()}")
print(f"Median strategy mean: {df_median[feature_names].mean().mean()}")
print(f"Mean strategy mean: {df_mean[feature_names].mean().mean()}")

print("\nSample missing values after filling:")
print(f"Zero strategy NaNs: {df_zero[feature_names].isna().sum().sum()}")
print(f"Median strategy NaNs: {df_median[feature_names].isna().sum().sum()}")
print(f"Mean strategy NaNs: {df_mean[feature_names].isna().sum().sum()}")


print(df_zero.shape)
print(df_median.shape)
print(df_mean.shape)

def evaluate_strategy(X, y, w):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        w_fold_train, w_fold_val = w[train_idx], w[val_idx]
        
        model = LGBMRegressor(random_state=42)
        model.fit(X_fold_train, y_fold_train, sample_weight=w_fold_train)
        y_pred = model.predict(X_fold_val)
        
        score = r2_score(y_fold_val, y_pred, sample_weight=w_fold_val)
        scores.append(score)
    
    return np.mean(scores)

# Convert DataFrames to numpy arrays
X_zero = df_zero[feature_names].to_numpy()
X_median = df_median[feature_names].to_numpy()
X_mean = df_mean[feature_names].to_numpy()

# Evaluate strategies
score_zero = evaluate_strategy(X_zero, y_train, w_train)
score_median = evaluate_strategy(X_median, y_train, w_train)
score_mean = evaluate_strategy(X_mean, y_train, w_train)

print(f"Weighted R² - Zero: {score_zero:.4f}")
print(f"Weighted R² - Median: {score_median:.4f}")
print(f"Weighted R² - Mean: {score_mean:.4f}")