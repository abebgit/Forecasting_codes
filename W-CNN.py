import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random
import os
import time
import psutil
import shap
import pywt

# Set seed for Reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

INSTALLED_CAPACITY = 1500.0
total_start_time = time.time()

# Random Search spaces
n_iter = 10

param_distributions = {
    'wavelet': ['db4', 'db6', 'sym4', 'sym6'],
    'filters': [32, 64, 96, 128],
    'kernel_size': [2, 3, 4, 5],
    'activation': ['relu', 'tanh'],
    'pool_size': [2, 3],
    'dropout_rate': [0.1, 0.2, 0.3],
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0005]
}

# Metrics functions
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

# Wavelet functions
def apply_wavelet_transform(series, wavelet='db4', level=1):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    approximation = coeffs[0]
    details = coeffs[1:]
    return approximation, details

def process_wavelet_data(X_data, wavelet='db4', level=1):
    approximations = []
    details_list = []
    for x in X_data:
        approximation, details = apply_wavelet_transform(x, wavelet=wavelet, level=level)
        approximations.append(approximation)
        details_list.append(details[0])
    approximations = np.array(approximations)
    details_list = np.array(details_list)
    X_wavelet = np.concatenate([approximations, details_list], axis=1)
    return X_wavelet, approximations, details_list

# Data Loading & Preprocessing
df = pd.read_excel('Training.xlsx')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)
df = df.ffill()

look_back = 12
for i in range(1, look_back + 1):
    df[f'power_lag{i}'] = df['power'].shift(i)
df.dropna(inplace=True)

feature_columns = ['speed', 'speed rate', 'temp', 'pitch'] + [f'power_lag{i}' for i in range(1, look_back + 1)]
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
y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1,1)).flatten()
y_test_scaled  = scaler_y.transform(y_test_raw.reshape(-1,1)).flatten()

# Validation split (80/20 chronological) on scaled data
train_size = int(len(X_train_scaled) * 0.8)
X_train_main_scaled = X_train_scaled[:train_size]
y_train_main_scaled = y_train_scaled[:train_size]
X_val_scaled = X_train_scaled[train_size:]
y_val_scaled = y_train_scaled[train_size:]

# Calculate MASE scaling factor
y_train_original = train_df['power'].values
naive_errors = np.abs(y_train_original[1:] - y_train_original[:-1])
mase_scaling_factor = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
print(f"MASE scaling factor (MAE of naive forecast on training): {mase_scaling_factor:.4f}")

# Random Search Loop
best_val_loss = float('inf')
best_params = None

print(f"Starting random search over {n_iter} random combinations...\n")
for i in range(n_iter):
    # Sample random hyperparameters
    combo = {
        'wavelet': random.choice(param_distributions['wavelet']),
        'filters': random.choice(param_distributions['filters']),
        'kernel_size': random.choice(param_distributions['kernel_size']),
        'activation': random.choice(param_distributions['activation']),
        'pool_size': random.choice(param_distributions['pool_size']),
        'dropout_rate': random.choice(param_distributions['dropout_rate']),
        'batch_size': random.choice(param_distributions['batch_size']),
        'learning_rate': random.choice(param_distributions['learning_rate'])
    }

    print(f"Iteration {i+1}/{n_iter} - Testing combination: {combo}")
    tf.keras.backend.clear_session()

    wavelet = combo['wavelet']
    X_train_wav, _, _ = process_wavelet_data(X_train_main_scaled, wavelet=wavelet, level=1)
    X_val_wav, _, _   = process_wavelet_data(X_val_scaled, wavelet=wavelet, level=1)

    def create_sequences(X, y, look_back):
        X_seq, y_seq = [], []
        for i in range(len(X) - look_back):
            X_seq.append(X[i:i+look_back])
            y_seq.append(y[i+look_back])
        return np.array(X_seq), np.array(y_seq)

    X_train_seq, y_train_seq = create_sequences(X_train_wav, y_train_main_scaled, look_back)
    X_val_seq, y_val_seq     = create_sequences(X_val_wav, y_val_scaled, look_back)

    # Build CNN model – use padding='same' to preserve sequence length until pooling
    model = Sequential()
    model.add(Conv1D(filters=combo['filters'], kernel_size=combo['kernel_size'],
                     activation=combo['activation'], padding='same',
                     input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(MaxPooling1D(pool_size=combo['pool_size']))
    model.add(Dropout(combo['dropout_rate']))
    model.add(Conv1D(filters=combo['filters'], kernel_size=combo['kernel_size'],
                     activation=combo['activation'], padding='same'))
    model.add(MaxPooling1D(pool_size=combo['pool_size']))
    model.add(Dropout(combo['dropout_rate']))
    model.add(Flatten())
    model.add(Dense(1))

    optimizer = Adam(learning_rate=combo['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=50,
        batch_size=combo['batch_size'],
        validation_data=(X_val_seq, y_val_seq),
        verbose=0,
        callbacks=[early_stopping]
    )

    val_loss = min(history.history['val_loss'])
    print(f"Validation loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = combo
        print("New best parameters found!")

print("\nRandom search completed.")
print(f"Best parameters: {best_params}")
print(f"Best validation loss: {best_val_loss}")

# Retrain best model with memory tracking
tf.keras.backend.clear_session()

wavelet = best_params['wavelet']
X_train_full_wav, approximations, details_list = process_wavelet_data(X_train_scaled, wavelet=wavelet, level=1)
X_val_full_wav, _, _   = process_wavelet_data(X_val_scaled, wavelet=wavelet, level=1)
X_test_full_wav, _, _  = process_wavelet_data(X_test_scaled, wavelet=wavelet, level=1)

X_train_full_seq, y_train_full_seq = create_sequences(X_train_full_wav, y_train_scaled, look_back)
X_val_full_seq, y_val_full_seq     = create_sequences(X_val_full_wav, y_val_scaled, look_back)
X_test_seq, y_test_seq             = create_sequences(X_test_full_wav, y_test_scaled, look_back)

best_model = Sequential([
    Conv1D(filters=best_params['filters'], kernel_size=best_params['kernel_size'],
           activation=best_params['activation'], padding='same',
           input_shape=(X_train_full_seq.shape[1], X_train_full_seq.shape[2])),
    MaxPooling1D(pool_size=best_params['pool_size']),
    Dropout(best_params['dropout_rate']),
    Conv1D(filters=best_params['filters'], kernel_size=best_params['kernel_size'],
           activation=best_params['activation'], padding='same'),
    MaxPooling1D(pool_size=best_params['pool_size']),
    Dropout(best_params['dropout_rate']),
    Flatten(),
    Dense(1)
])

optimizer = Adam(learning_rate=best_params['learning_rate'])
best_model.compile(optimizer=optimizer, loss='mean_squared_error')

trainable_params = int(np.sum([np.prod(v.shape) for v in best_model.trainable_weights]))
total_params = int(best_model.count_params())


import tracemalloc


process = psutil.Process(os.getpid())

print("\n🚀 Training best model with full memory diagnostics...")

# Track memory before training
memory_before = process.memory_info().rss / (1024 ** 2)

# Start tracemalloc for peak memory tracking
tracemalloc.start()

train_start = time.time()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_best = best_model.fit(
    X_train_full_seq, y_train_full_seq,
    epochs=100,
    batch_size=best_params['batch_size'],
    validation_data=(X_val_full_seq, y_val_full_seq),
    verbose=1,
    callbacks=[early_stopping]
)

train_end = time.time()

# Track memory after training
memory_after = process.memory_info().rss / (1024 ** 2)

# Track Peack memory during training
current, peak = tracemalloc.get_traced_memory()
peak_memory_mb = peak / (1024 ** 2)

tracemalloc.stop()

# Derived metrics
training_duration = train_end - train_start
epochs_ran = len(history_best.history.get('loss', []))
seconds_per_epoch = training_duration / epochs_ran if epochs_ran > 0 else float("nan")

memory_increase_mb = memory_after - memory_before

# Print results
print(f"\n✅ Training completed in {training_duration:.2f} seconds")
print(f"   • Epochs run: {epochs_ran}")
print(f"   • Seconds per epoch: {seconds_per_epoch:.4f} s/epoch")

print("\n📊 MEMORY USAGE SUMMARY")
print(f"   • Memory before training:  {memory_before:.2f} MB")
print(f"   • Memory after training:   {memory_after:.2f} MB")
print(f"   • Actual memory increase:  {memory_increase_mb:.2f} MB")
print(f"   • Peak memory during training: {peak_memory_mb:.2f} MB (via tracemalloc)")

print(f"\n🔢 Trainable Params: {trainable_params:,} | Total Params: {total_params:,}")

# Big O Notation Estimates
def estimate_complexity(model, X_shape, batch_size, epochs, best_params):
    batch, steps, features = X_shape
    flops_per_epoch = 0
    filters = best_params['filters']
    kernel = best_params['kernel_size']
    pool = best_params['pool_size']
    # First Conv: O(batch * steps * filters * kernel * features)
    flops_conv1 = batch * steps * filters * kernel * features
    flops_per_epoch += flops_conv1
    # After first pool, steps become ceil(steps/pool) – we approximate as steps // pool + 1
    steps2 = (steps + pool - 1) // pool   # ceil division
    # Second Conv: input features = filters
    flops_conv2 = batch * steps2 * filters * kernel * filters
    flops_per_epoch += flops_conv2
    # After second pool, steps3 = ceil(steps2/pool)
    steps3 = (steps2 + pool - 1) // pool
    # Dense layer: flatten size = steps3 * filters
    flops_dense = batch * (steps3 * filters) * 1
    flops_per_epoch += flops_dense

    n_batches_per_epoch = X_train_full_seq.shape[0] // batch_size
    total_flops_training = flops_per_epoch * epochs * n_batches_per_epoch
    inference_flops_per_sample = flops_per_epoch / batch

    return {
        'Big O Training': f"O(epochs * n_samples * (filters * kernel * features))",
        'Approx FLOPs per epoch': f"{flops_per_epoch:.2e}",
        'Approx total training FLOPs': f"{total_flops_training:.2e}",
        'Inference FLOPs per sample': f"{inference_flops_per_sample:.2e}",
        'Space complexity': f"O(filters * kernel * features)"
    }

complexity = estimate_complexity(best_model, X_train_full_seq.shape, best_params['batch_size'], epochs_ran, best_params)

print("\n🧮 Big O Complexity Estimates:")
for k, v in complexity.items():
    print(f"{k}: {v}")

# Evaluate on test dataset
inference_start = time.time()
y_pred_scaled = best_model.predict(X_test_seq, verbose=0)
inference_end = time.time()
inference_time_per_sample = (inference_end - inference_start) / len(X_test_seq) if len(X_test_seq) > 0 else np.nan

y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_test_original = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

# Standard metrics
mae = mean_absolute_error(y_test_original, y_pred)
mape = mean_absolute_percentage_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)
mbe = mean_bias_error(y_test_original, y_pred)

# Relative error-based metrics
eps = 1e-8
relative_error = (y_test_original - y_pred) / (y_test_original + eps)
mare = np.mean(np.abs(relative_error))
msre = np.mean(relative_error ** 2)
rmsre = np.sqrt(msre)
rmspe = rmsre * 100

# sMAPE and MASE
smape = symmetric_mean_absolute_percentage_error(y_test_original, y_pred)
mase = mean_absolute_scaled_error(y_test_original, y_pred, mase_scaling_factor)

nMAE = mae / INSTALLED_CAPACITY
nRMSE = rmse / INSTALLED_CAPACITY
nMBE = mbe / INSTALLED_CAPACITY

print('\nBest Model Test Performance:')
print(f"MAE: {mae:.4f} | nMAE: {nMAE:.6f}")
print(f"MAPE: {mape:.2f}% | RMSE: {rmse:.4f} | nRMSE: {nRMSE:.6f}")
print(f"MBE: {mbe:.4f} | nMBE: {nMBE:.6f}")
print(f"R²: {r2:.4f}")
print(f"MARE: {mare:.6f} | MSRE: {msre:.6f}")
print(f"RMSRE: {rmsre:.6f} | RMSPE: {rmspe:.2f}%")
print(f"sMAPE: {smape:.2f}% | MASE: {mase:.4f}")
print(f"Inference Time per Sample: {inference_time_per_sample*1000:.4f} ms")

# Predict the next day
next_day_data = pd.read_excel('Input_to_be_forecasted.xlsx')

for i in range(1, look_back + 1):
    next_day_data[f'power_lag{i}'] = next_day_data['power'].shift(i)
# Drop the first look_back rows
next_day_data.dropna(subset=[f'power_lag{i}' for i in range(1, look_back + 1)], inplace=True)
timestamps = next_day_data['Timestamp']
X_next_raw = next_day_data[feature_columns].values
y_next_actual = next_day_data['power'].values

X_next_scaled = scaler_X.transform(X_next_raw)
X_next_wav, _, _ = process_wavelet_data(X_next_scaled, wavelet=wavelet, level=1)

X_next_seq, _ = create_sequences(X_next_wav, np.zeros(len(X_next_wav)), look_back)

next_day_pred_start = time.time()
y_next_scaled = best_model.predict(X_next_seq)
next_day_pred_end = time.time()
next_day_prediction_time = next_day_pred_end - next_day_pred_start

y_next_forecast = scaler_y.inverse_transform(y_next_scaled).flatten()
forecast_indices = range(look_back, len(timestamps))
df_forecast = pd.DataFrame({
    'Time': timestamps.iloc[forecast_indices],
    'Actual_Power': y_next_actual[forecast_indices],
    'Forecasted_Power': y_next_forecast
})

# Metrics for Next-day prediction
mae_next_day = mean_absolute_error(df_forecast['Actual_Power'], df_forecast['Forecasted_Power'])
mape_next_day = mean_absolute_percentage_error(df_forecast['Actual_Power'], df_forecast['Forecasted_Power'])
rmse_next_day = np.sqrt(mean_squared_error(df_forecast['Actual_Power'], df_forecast['Forecasted_Power']))
r2_next_day = r2_score(df_forecast['Actual_Power'], df_forecast['Forecasted_Power'])
mbe_next_day = mean_bias_error(df_forecast['Actual_Power'], df_forecast['Forecasted_Power'])

rel_error_next = (df_forecast['Actual_Power'].values - df_forecast['Forecasted_Power'].values) / (df_forecast['Actual_Power'].values + eps)
mare_next = np.mean(np.abs(rel_error_next))
msre_next = np.mean(rel_error_next ** 2)
rmsre_next = np.sqrt(msre_next)
rmspe_next = rmsre_next * 100

smape_next = symmetric_mean_absolute_percentage_error(df_forecast['Actual_Power'], df_forecast['Forecasted_Power'])
mase_next = mean_absolute_scaled_error(df_forecast['Actual_Power'], df_forecast['Forecasted_Power'], mase_scaling_factor)

nMAE_next_day = mae_next_day / INSTALLED_CAPACITY
nRMSE_next_day = rmse_next_day / INSTALLED_CAPACITY
nMBE_next_day = mbe_next_day / INSTALLED_CAPACITY

total_time = time.time() - total_start_time

print('\n🔮 Best Model Next-Day Forecast Performance:')
print(f"MAE: {mae_next_day:.4f} | nMAE: {nMAE_next_day:.6f}")
print(f"MAPE: {mape_next_day:.2f}% | RMSE: {rmse_next_day:.4f} | nRMSE: {nRMSE_next_day:.6f}")
print(f"MBE: {mbe_next_day:.4f} | nMBE: {nMBE_next_day:.6f}")
print(f"R²: {r2_next_day:.4f}")
print(f"MARE: {mare_next:.6f} | MSRE: {msre_next:.6f}")
print(f"RMSRE: {rmsre_next:.6f} | RMSPE: {rmspe_next:.2f}%")
print(f"sMAPE: {smape_next:.2f}% | MASE: {mase_next:.4f}")
print(f"Next-Day Prediction Time: {next_day_prediction_time:.4f} s")
print(f"Total Script Execution Time: {total_time:.2f} s")

# Export results to excel
model_info = pd.DataFrame({
    'Model': ['Wavelet-CNN'],
    'Wavelet': [best_params['wavelet']],
    'Filters': [best_params['filters']],
    'Kernel Size': [best_params['kernel_size']],
    'Activation': [best_params['activation']],
    'Pool Size': [best_params['pool_size']],
    'Trainable Params': [trainable_params],
    'Total Params': [total_params],
    'Batch Size': [best_params['batch_size']],
    'Look-back Window': [look_back],
    'Optimizer': ['Adam'],
    'Learning Rate': [best_params['learning_rate']],
    'Dropout Rate': [best_params['dropout_rate']],
    'Epochs Ran': [epochs_ran],
    'Memory Increase (MB)': [memory_increase_mb]
})

timing_summary = pd.DataFrame({
    'Metric': ['Training Time (s)', 'Seconds per Epoch (s)', 'Test Prediction Time (s)', 'Next Day Prediction Time (s)', 'Total Execution Time (s)'],
    'Value': [training_duration, seconds_per_epoch, inference_end - inference_start, next_day_prediction_time, total_time]
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
    'MASE': [mase],
    'Inference Time per Sample (ms)': [inference_time_per_sample * 1000]
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

complexity_df = pd.DataFrame([complexity])

with pd.ExcelWriter('W-CNN_randomsearch_best_model.xlsx') as writer:
    model_info.to_excel(writer, sheet_name='Model_Info', index=False)
    timing_summary.to_excel(writer, sheet_name='Timing_Summary', index=False)
    test_metrics.to_excel(writer, sheet_name='Test_Performance', index=False)
    nextday_metrics.to_excel(writer, sheet_name='NextDay_Performance', index=False)
    df_forecast.to_excel(writer, sheet_name='NextDay_Forecast', index=False)
    complexity_df.to_excel(writer, sheet_name='BigO_Estimates', index=False)

print("\n✅ All metrics + forecasts + Big O exported to 'W-CNN_randomsearch_best_model.xlsx'")

# ---------------------- Plotting ----------------------
plt.figure(figsize=(12, 6))
plt.plot(df_forecast['Time'], df_forecast['Actual_Power'], label='Actual Power', marker='o', markersize=6, linewidth=2, color='blue')
plt.plot(df_forecast['Time'], df_forecast['Forecasted_Power'], label='Forecasted Power', marker='x', markersize=6, linewidth=2, color='red')
plt.xlabel('Time', fontsize=14, fontweight='bold')
plt.ylabel('Power Output (kW)', fontsize=14, fontweight='bold')
plt.title('Turbine-Next Day Forecast vs Actual (Wavelet-CNN)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("WaveletCNN_forecast_plot.pdf", dpi=500)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history_best.history['loss'], label='Training Loss')
plt.plot(history_best.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - Wavelet-CNN')
plt.legend()
plt.grid(True)
plt.savefig("WaveletCNN_training_loss.pdf")
plt.show()

# SHAP analysis
print("Starting SHAP analysis...")
try:
    n_approx = approximations.shape[1]
    n_detail = details_list.shape[1]
    wavelet_feature_names = [f'approx_{i}' for i in range(n_approx)] + [f'detail_{i}' for i in range(n_detail)]

    background_size = min(50, X_train_full_seq.shape[0])
    background_indices = np.random.choice(X_train_full_seq.shape[0], background_size, replace=False)
    background_data = X_train_full_seq[background_indices]

    test_sample_size = min(50, X_test_seq.shape[0])
    test_indices = np.random.choice(X_test_seq.shape[0], test_sample_size, replace=False)
    X_test_sample = X_test_seq[test_indices]

    X_test_flat = X_test_sample.reshape(X_test_sample.shape[0], -1)
    background_flat = background_data.reshape(background_data.shape[0], -1)

    try:
        explainer = shap.DeepExplainer(best_model, background_data)
        shap_out = explainer.shap_values(X_test_sample)
        shap_values = shap_out[0] if isinstance(shap_out, list) else shap_out
        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[0]
        print("Used DeepExplainer for SHAP.")
    except Exception as de:
        print(f"DeepExplainer failed: {de}. Falling back to KernelExplainer (slower).")
        def predict_fn_flat(data_2d):
            data_3d = data_2d.reshape((data_2d.shape[0], X_train_full_seq.shape[1], X_train_full_seq.shape[2]))
            return best_model.predict(data_3d, verbose=0).flatten()
        explainer = shap.KernelExplainer(predict_fn_flat, background_flat)
        shap_values = explainer.shap_values(X_test_flat, nsamples=200)
        expected_value = explainer.expected_value
        print("Used KernelExplainer for SHAP (fallback).")

    shap_values = np.array(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_flat, feature_names=wavelet_feature_names, plot_type='bar', show=False)
    plt.title('Turbine-Feature Importance (SHAP) - Wavelet-CNN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Turbine-shap_feature_importance.pdf')
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_flat, feature_names=wavelet_feature_names, show=False)
    plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Turbine-shap_detailed_plot.pdf')
    plt.show()

    if isinstance(expected_value, np.ndarray):
        ev = float(expected_value) if expected_value.size == 1 else expected_value[0]
    else:
        ev = float(expected_value)

    try:
        fp = shap.force_plot(ev, shap_values[0], X_test_flat[0], feature_names=wavelet_feature_names, matplotlib=False)
        shap.save_html("shap_force_plot_sample0.html", fp)
        print("Saved interactive force plot to shap_force_plot_sample0.html")
    except Exception as fp_e:
        print(f"Interactive force plot failed ({fp_e}), trying matplotlib version.")
        try:
            shap.force_plot(ev, shap_values[0], X_test_flat[0], feature_names=wavelet_feature_names, matplotlib=True)
            plt.tight_layout()
            plt.savefig('shap_force_plot_sample0.png')
            plt.show()
            print("Saved matplotlib force plot to shap_force_plot_sample0.png")
        except Exception as plt_e:
            print(f"Matplotlib force plot also failed: {plt_e}")

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.flatten()
    feature_importance = pd.DataFrame({
        'Feature': wavelet_feature_names,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance based on SHAP values:")
    print(feature_importance)

except Exception as e:
    print(f"Error in SHAP calculation: {e}")
    print("Continuing without SHAP visualization...")

print("\nAll visualizations")