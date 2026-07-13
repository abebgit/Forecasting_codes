import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import random
import os
import time
import psutil
import shap

# ---------------------- Reproducibility ----------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

INSTALLED_CAPACITY = 1500.0
total_start_time = time.time()

# ---------------------- Helper Functions ----------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = np.finfo(float).eps
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, eps, y_true))) * 100

def mean_bias_error(y_true, y_pred):
    return np.mean(np.array(y_pred) - np.array(y_true))

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    eps = 1e-8
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / denominator)

def mean_absolute_scaled_error(y_true, y_pred, scaling_factor):
    return mean_absolute_error(y_true, y_pred) / scaling_factor

def mase(y_true, y_pred, m=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae_forecast = np.mean(np.abs(y_true - y_pred))
    mae_naive = np.mean(np.abs(y_true[m:] - y_true[:-m]))
    return mae_forecast / mae_naive

# ---------------------- Data Loading & Preprocessing ----------------------
df = pd.read_excel('Training.xlsx')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)
df = df.ffill()

look_back = 12
for i in range(1, look_back + 1):
    df[f'power_lag{i}'] = df['power'].shift(i)
df.dropna(inplace=True)

feature_columns = ['speed', 'speed rate','temp', 'pitch'] + [f'power_lag{i}' for i in range(1, look_back + 1)]
target_column = 'power'
df['year'] = df['Timestamp'].dt.year

# Chronological split
train_df = df[df['year'] <= 2022]
test_df  = df[df['year'] == 2023]

X_train_raw = train_df[feature_columns].values
y_train_raw = train_df[target_column].values
X_test_raw  = test_df[feature_columns].values
y_test_raw  = test_df[target_column].values

# Scaling (fit only on training)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled  = scaler_X.transform(X_test_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

# Validation split (80/20 chronological) on the scaled data
train_size = int(len(X_train_scaled) * 0.8)
X_train_main = X_train_scaled[:train_size]
y_train_main = y_train_scaled[:train_size]
X_val = X_train_scaled[train_size:]
y_val = y_train_scaled[train_size:]

# ---------------------- Compute MASE Scaling Factor ----------------------
y_train_original = train_df['power'].values
naive_errors = np.abs(y_train_original[1:] - y_train_original[:-1])
mase_scaling_factor = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
print(f"MASE scaling factor (MAE of naive forecast on training): {mase_scaling_factor:.4f}")

# ---------------------- Hyperparameter Distributions ----------------------
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8, 9],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 1.5, 2.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4]
}

# ---------------------- Random Search ----------------------
print("Performing hyperparameter tuning with RandomSearch...")

# Remove early_stopping_rounds from the estimator used in search
xgb = XGBRegressor(random_state=seed, verbosity=0)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=100,                # number of random combinations
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    random_state=seed,
    n_jobs=-1
)

# Fit on training data (no eval_set needed)
random_search.fit(X_train_main, y_train_main)

print("Best parameters found: ", random_search.best_params_)
print("Best score (neg MSE): ", random_search.best_score_)

best_params = random_search.best_params_

# ---------------------- Retrain Best Model with Memory Tracking ----------------------
print("\n🚀 Training best model with memory tracking...")
train_start = time.time()
memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

# Now add early stopping for final training
final_model = XGBRegressor(
    **best_params,
    random_state=seed,
    early_stopping_rounds=20,
    verbosity=1
)

final_model.fit(
    X_train_scaled, y_train_scaled,
    eval_set=[(X_val, y_val)],
    verbose=True
)

train_end = time.time()
memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

training_duration = train_end - train_start
memory_used_mb = memory_after - memory_before

# Approximate parameter count
n_estimators = best_params.get('n_estimators', 100)
max_depth = best_params.get('max_depth', 6)
approx_params = n_estimators * (2 ** (max_depth + 1) - 1)

print(f"\n✅ Training Completed in {training_duration:.2f} seconds")
print(f"📊 Memory increase during training: {memory_used_mb:.2f} MB")
print(f"🔢 Approx. parameters: {approx_params:,}")

# ---------------------- Evaluation on Test Set ----------------------
inference_start = time.time()
y_pred_scaled = final_model.predict(X_test_scaled)
inference_end = time.time()
inference_time_per_sample = (inference_end - inference_start) / len(X_test_scaled) if len(X_test_scaled) > 0 else np.nan

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_original = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# ----- Standard metrics -----
mae = mean_absolute_error(y_test_original, y_pred)
mape = mean_absolute_percentage_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)
mbe = mean_bias_error(y_test_original, y_pred)

# ----- Relative error based metrics -----
eps = 1e-8
relative_error = (y_test_original - y_pred) / (y_test_original + eps)
mare = np.mean(np.abs(relative_error))
msre = np.mean(relative_error ** 2)
rmsre = np.sqrt(msre)
rmspe = rmsre * 100

# ----- sMAPE and MASE -----
smape = symmetric_mean_absolute_percentage_error(y_test_original, y_pred)
mase_value = mean_absolute_scaled_error(y_test_original, y_pred, mase_scaling_factor)

nMAE = mae / INSTALLED_CAPACITY
nRMSE = rmse / INSTALLED_CAPACITY
nMBE = mbe / INSTALLED_CAPACITY

print('\n📈 Best Model Test Performance-random:')
print(f"MAE: {mae:.4f} | nMAE: {nMAE:.6f}")
print(f"MAPE: {mape:.2f}% | RMSE: {rmse:.4f} | nRMSE: {nRMSE:.6f}")
print(f"MBE: {mbe:.4f} | nMBE: {nMBE:.6f}")
print(f"R²: {r2:.4f}")
print(f"MARE: {mare:.6f} | MSRE: {msre:.6f}")
print(f"RMSRE: {rmsre:.6f} | RMSPE: {rmspe:.2f}%")
print(f"sMAPE: {smape:.2f}% | MASE: {mase_value:.4f}")
print(f"Inference Time per Sample: {inference_time_per_sample*1000:.4f} ms")

# ---------------------- Next-Day Forecasting ----------------------

# Load Excel and remove fully empty rows at the bottom
next_day_data = pd.read_excel('Input_to_be_predicted.xlsx').dropna(how='all')

# Set your look-back window
look_back = 12

# Create lag features
for i in range(1, look_back + 1):
    next_day_data[f'power_lag{i}'] = next_day_data['power'].shift(i)

# ✅ Drop ONLY the first look_back rows (those created by shifting)
next_day_data = next_day_data.iloc[look_back:].reset_index(drop=True)

# Show result
print(next_day_data.head())
print(next_day_data.tail())

timestamps = next_day_data['Timestamp']
X_next_raw = next_day_data[feature_columns].values
y_next_actual = next_day_data['power'].values

X_next_scaled = scaler_X.transform(X_next_raw)

next_day_pred_start = time.time()
y_next_scaled = final_model.predict(X_next_scaled)
next_day_pred_end = time.time()
next_day_prediction_time = next_day_pred_end - next_day_pred_start

y_next_forecast = scaler_y.inverse_transform(y_next_scaled.reshape(-1, 1)).flatten()

df_forecast = pd.DataFrame({
    'Time': timestamps,
    'Actual_Power': y_next_actual,
    'Forecasted_Power': y_next_forecast
})

# ----- Next-day metrics -----
mae_next_day = mean_absolute_error(y_next_actual, y_next_forecast)
mape_next_day = mean_absolute_percentage_error(y_next_actual, y_next_forecast)
rmse_next_day = np.sqrt(mean_squared_error(y_next_actual, y_next_forecast))
r2_next_day = r2_score(y_next_actual, y_next_forecast)
mbe_next_day = mean_bias_error(y_next_actual, y_next_forecast)

rel_error_next = (y_next_actual - y_next_forecast) / (y_next_actual + eps)
mare_next = np.mean(np.abs(rel_error_next))
msre_next = np.mean(rel_error_next ** 2)
rmsre_next = np.sqrt(msre_next)
rmspe_next = rmsre_next * 100

smape_next = symmetric_mean_absolute_percentage_error(y_next_actual, y_next_forecast)
mase_next = mean_absolute_scaled_error(y_next_actual, y_next_forecast, mase_scaling_factor)

nMAE_next_day = mae_next_day / INSTALLED_CAPACITY
nRMSE_next_day = rmse_next_day / INSTALLED_CAPACITY
nMBE_next_day = mbe_next_day / INSTALLED_CAPACITY

total_time = time.time() - total_start_time

print('\n🔮 Best Model Next-Day Forecast Performance-Random-xgboost:')
print(f"MAE: {mae_next_day:.4f} | nMAE: {nMAE_next_day:.6f}")
print(f"MAPE: {mape_next_day:.2f}% | RMSE: {rmse_next_day:.4f} | nRMSE: {nRMSE_next_day:.6f}")
print(f"MBE: {mbe_next_day:.4f} | nMBE: {nMBE_next_day:.6f}")
print(f"R²: {r2_next_day:.4f}")
print(f"MARE: {mare_next:.6f} | MSRE: {msre_next:.6f}")
print(f"RMSRE: {rmsre_next:.6f} | RMSPE: {rmspe_next:.2f}%")
print(f"sMAPE: {smape_next:.2f}% | MASE: {mase_next:.4f}")
print(f"Next-Day Prediction Time: {next_day_prediction_time:.4f} s")
print(f"Total Script Execution Time: {total_time:.2f} s")

# ---------------------- Export All Results to Excel ----------------------
model_info = pd.DataFrame({
    'Model': ['XGBoost'],
    'Best Params': [str(best_params)],
    'Approx Params': [approx_params],
    'Memory Increase (MB)': [memory_used_mb],
    'Training Time (s)': [training_duration],
    'Inference Time per Sample (ms)': [inference_time_per_sample * 1000]
})

timing_summary = pd.DataFrame({
    'Metric': ['Training Time (s)', 'Test Prediction Time (s)', 'Next Day Prediction Time (s)', 'Total Execution Time (s)'],
    'Value': [training_duration, inference_end - inference_start, next_day_prediction_time, total_time]
})

test_metrics = pd.DataFrame({
    'Dataset': ['Test Set'],
    'MAE': [mae],
    'nMAE': [nMAE],
    'RMSE': [rmse],
    'nRMSE': [nRMSE],
    'MBE': [mbe],
    'nMBE': [nMBE],
    'MAPE (%)': [mape],
    'R²': [r2],
    'MARE': [mare],
    'MSRE': [msre],
    'RMSRE': [rmsre],
    'RMSPE (%)': [rmspe],
    'sMAPE (%)': [smape],
    'MASE': [mase_value]
})

nextday_metrics = pd.DataFrame({
    'Dataset': ['Next Day Forecast'],
    'MAE': [mae_next_day],
    'nMAE': [nMAE_next_day],
    'RMSE': [rmse_next_day],
    'nRMSE': [nRMSE_next_day],
    'MBE': [mbe_next_day],
    'nMBE': [nMBE_next_day],
    'MAPE (%)': [mape_next_day],
    'R²': [r2_next_day],
    'MARE': [mare_next],
    'MSRE': [msre_next],
    'RMSRE': [rmsre_next],
    'RMSPE (%)': [rmspe_next],
    'sMAPE (%)': [smape_next],
    'MASE': [mase_next],
    'Next Day Prediction Time (s)': [next_day_prediction_time]
})

with pd.ExcelWriter('Turbine-XGBoost_randomsearch_best_model.xlsx') as writer:
    model_info.to_excel(writer, sheet_name='Model_Info', index=False)
    timing_summary.to_excel(writer, sheet_name='Timing_Summary', index=False)
    test_metrics.to_excel(writer, sheet_name='Test_Performance', index=False)
    nextday_metrics.to_excel(writer, sheet_name='NextDay_Performance', index=False)
    df_forecast.to_excel(writer, sheet_name='NextDay_Forecast', index=False)

print("\n✅ All metrics + forecasts exported to 'N73-XGBoost_randomsearch_best_model-Random-raw.xlsx'")

# ---------------------- Plotting ----------------------


print("\nAll visualizations have been saved. Script completed successfully!")
final_model = XGBRegressor(
    **best_params,
    random_state=seed,
    early_stopping_rounds=20,
    verbosity=1
)

# Include BOTH train and validation to see both curves

final_model.fit(
    X_train_scaled, y_train_scaled,
    eval_set=[(X_train_scaled, y_train_scaled),
              (X_val, y_val)],
    verbose=False
)

# ---------------------- Plot XGBoost Training & Validation Loss ----------------------
# xgboost.sklearn API supports evals_result()
try:
    evals = final_model.evals_result()  # dict like {'train': {'rmse': [...]}, 'valid': {'rmse': [...]}}

    # Compatible access (handles different xgboost versions / key names)
    # Prefer explicit names if available, else fall back to validation_*
    def get_curve(e, name_candidates):
        for n in name_candidates:
            if n in e:
                # pick first metric recorded (rmse if eval_metric='rmse')
                metrics_dict = e[n]
                if isinstance(metrics_dict, dict) and len(metrics_dict) > 0:
                    first_metric = list(metrics_dict.keys())[0]
                    return metrics_dict[first_metric]
        return []

    train_rmse = get_curve(evals, ["train", "validation_0"])
    val_rmse   = get_curve(evals, ["valid", "validation_1", "eval", "validation"])

    if len(train_rmse) == 0 and len(val_rmse) == 0:
        print("[XGB] No eval curves found in evals_result(); check eval_set / eval_metric.")
    else:
        rounds = range(1, max(len(train_rmse), len(val_rmse)) + 1)
        plt.figure(figsize=(10, 5))
        if len(train_rmse) > 0:
            plt.plot(rounds[:len(train_rmse)], train_rmse, label='Train RMSE (scaled)', color='#1f77b4', linewidth=2)
        if len(val_rmse) > 0:
            plt.plot(rounds[:len(val_rmse)],   val_rmse,   label='Validation RMSE (scaled)', color='#d62728', linewidth=2)
        plt.xlabel('Boosting Rounds', fontsize=12)
        plt.ylabel('RMSE (target is MinMax scaled)', fontsize=12)
        plt.title('XGBoost Training vs Validation Loss', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig('XGBoost_training_validation_loss.pdf', dpi=300)
        plt.show()

    # Optional: report best iteration/score from early stopping
    if hasattr(final_model, "best_iteration"):
        print(f"Best iteration (0-based): {final_model.best_iteration}")
    if hasattr(final_model, "best_score"):
        print(f"Best validation RMSE (scaled): {final_model.best_score:.6f}")

except Exception as e:
    print(f"[XGB] Could not plot training/validation curves: {e}")
# ---------------------- SHAP Analysis (TreeExplainer for XGBoost) ----------------------
print("Starting SHAP analysis...")
try:
    # Background sample from training for SHAP
    background_size = min(200, X_train_main.shape[0])
    background_indices = np.random.choice(X_train_main.shape[0], background_size, replace=False)
    background_data = X_train_main[background_indices]

    # Test sample for SHAP visualization
    test_sample_size = min(200, X_test_scaled.shape[0])
    test_indices = np.random.choice(X_test_scaled.shape[0], test_sample_size, replace=False)
    X_test_sample = X_test_scaled[test_indices]

    # TreeExplainer is the correct one for XGBoost trees
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_sample)

    # Summary bar plot (mean |SHAP|)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_columns, plot_type='bar', show=False)
    plt.title('Feature Importance (SHAP) - XGBoost', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('SHAP_feature_importance_XGBoost.pdf')
    plt.show()

    # Detailed summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_columns, show=False)
    plt.title('SHAP Summary Plot - XGBoost', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('SHAP_summary_XGBoost- (more features).pdf')
    plt.show()

    # Optional: one force plot (may require JS in notebooks; fallback handled)
    expected_value = explainer.expected_value
    try:
        fp = shap.force_plot(expected_value, shap_values[0], X_test_sample[0], feature_names=feature_columns, matplotlib=False)
        shap.save_html("SHAP_force_plot_sample0_XGBoost-Random.html", fp)
        print("Saved interactive force plot to N74-SHAP_force_plot_sample0_XGBoost-Fixed.html")
    except Exception as fp_e:
        print(f"Interactive force plot failed ({fp_e}), trying matplotlib version.")
        try:
            shap.force_plot(expected_value, shap_values[0], X_test_sample[0], feature_names=feature_columns, matplotlib=True)
            plt.tight_layout()
            plt.savefig('SHAP_force_plot_sample0_XGBoost-random.p')
            plt.show()
            print("Saved matplotlib force plot to N73-SHAP_force_plot_sample0_XGBoost.pdf")
        except Exception as plt_e:
            print(f"Matplotlib force plot also failed: {plt_e}")

    # Print mean |SHAP| per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance based on mean |SHAP| values:")
    print(feature_importance)

except Exception as e:
    print(f"Error in SHAP calculation: {e}")
    print("Continuing without SHAP visualization...")

print("\nAll visualizations have been saved. Script completed successfully!")
