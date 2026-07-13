import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random
import os
import time
import psutil
import shap
import tracemalloc

# Set seed for reproducability
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
    'cnn_filters': [32, 64, 96, 128],
    'cnn_kernel_size': [3, 5],
    'lstm_units': [32, 64, 96, 128],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0005],
    'batch_size': [16, 32]
}

# Accuracy metrics functions
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

# Data Loading & Preprocessing
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

# Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train_raw)
X_test  = scaler_X.transform(X_test_raw)
y_train = scaler_y.fit_transform(y_train_raw.reshape(-1,1)).flatten()
y_test  = scaler_y.transform(y_test_raw.reshape(-1,1)).flatten()

# Validation split (80/20 chronological)
train_size = int(len(X_train) * 0.8)
X_train_main = X_train[:train_size]
y_train_main = y_train[:train_size]
X_val = X_train[train_size:]
y_val = y_train[train_size:]

# Create sliding windows
def create_sequences(X, y, look_back):
    X_seq, y_seq = [], []
    for i in range(len(X) - look_back):
        X_seq.append(X[i:i+look_back])
        y_seq.append(y[i+look_back])
    return np.array(X_seq), np.array(y_seq)

X_train_main, y_train_main = create_sequences(X_train_main, y_train_main, look_back)
X_val, y_val               = create_sequences(X_val, y_val, look_back)
X_test, y_test             = create_sequences(X_test, y_test, look_back)

print("Train shape:", X_train_main.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# Compute MASE Scaling Factor
y_train_original = train_df['power'].values
naive_errors = np.abs(y_train_original[1:] - y_train_original[:-1])
mase_scaling_factor = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
print(f"MASE scaling factor (MAE of naive forecast on training): {mase_scaling_factor:.4f}")

# Random Search Loop
best_val_loss = float('inf')
best_params = None

print(f"Starting random search over {n_iter} random combinations...\n")
for i in range(n_iter):
    combo = {
        'cnn_filters': random.choice(param_distributions['cnn_filters']),
        'cnn_kernel_size': random.choice(param_distributions['cnn_kernel_size']),
        'lstm_units': random.choice(param_distributions['lstm_units']),
        'dropout_rate': random.choice(param_distributions['dropout_rate']),
        'batch_size': random.choice(param_distributions['batch_size']),
        'learning_rate': random.choice(param_distributions['learning_rate'])
    }

    print(f"Iteration {i+1}/{n_iter} - Testing combination: {combo}")
    tf.keras.backend.clear_session()

    model = Sequential()
    model.add(Conv1D(filters=combo['cnn_filters'], kernel_size=combo['cnn_kernel_size'],
                     activation='relu', input_shape=(X_train_main.shape[1], X_train_main.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(combo['dropout_rate']))
    model.add(LSTM(combo['lstm_units']))
    model.add(Dropout(combo['dropout_rate']))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=combo['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

    history = model.fit(
        X_train_main, y_train_main,
        epochs=100,
        batch_size=combo['batch_size'],
        validation_data=(X_val, y_val),
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

# Retrain the Best Model with Memory Tracking
tf.keras.backend.clear_session()

best_model = Sequential([
    Conv1D(filters=best_params['cnn_filters'], kernel_size=best_params['cnn_kernel_size'],
           activation='relu', input_shape=(X_train_main.shape[1], X_train_main.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(best_params['dropout_rate']),
    LSTM(best_params['lstm_units'], return_sequences=True),
    Dropout(best_params['dropout_rate']),
    LSTM(best_params['lstm_units']),
    Dropout(best_params['dropout_rate']),
    Dense(1)
])

optimizer = Adam(learning_rate=best_params['learning_rate'])
best_model.compile(optimizer=optimizer, loss='mean_squared_error')

trainable_params = int(np.sum([np.prod(v.shape) for v in best_model.trainable_weights]))
total_params = int(best_model.count_params())

print("\n🚀 Training best model with memory tracking...")

process = psutil.Process(os.getpid())

print("\nTraining best model with full memory diagnostics...")

# Track Memory before training
memory_before = process.memory_info().rss / (1024 ** 2)

# Start tracemalloc for peak memory tracking
tracemalloc.start()

train_start = time.time()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_best = best_model.fit(
    X_train_main, y_train_main,
    epochs=100,
    batch_size=best_params['batch_size'],
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stopping]
)
train_end = time.time()

# Track memory after training
memory_after = process.memory_info().rss / (1024 ** 2)

# Track peak memory during training
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

#  Big O Notation Estimates
def estimate_complexity(model, X_shape, batch_size, epochs, best_params):
    batch, steps, features = X_shape
    flops_per_epoch = 0

    conv_filters = best_params['cnn_filters']
    kernel_size = best_params['cnn_kernel_size']
    flops_conv = batch * steps * conv_filters * kernel_size * features
    flops_per_epoch += flops_conv

    lstm_units = best_params['lstm_units']
    flops_lstm1 = batch * steps * (lstm_units ** 2)
    flops_per_epoch += flops_lstm1
    flops_lstm2 = batch * steps * (lstm_units ** 2)
    flops_per_epoch += flops_lstm2

    flops_dense = batch * steps * 1
    flops_per_epoch += flops_dense

    n_batches_per_epoch = X_train_main.shape[0] // batch_size
    total_flops_training = flops_per_epoch * epochs * n_batches_per_epoch
    inference_flops_per_sample = flops_per_epoch / batch

    return {
        'Big O Training': f"O(epochs * n_samples * (LSTM_units^2 + conv_filters * kernel_size * features))",
        'Approx FLOPs per epoch': f"{flops_per_epoch:.2e}",
        'Approx total training FLOPs': f"{total_flops_training:.2e}",
        'Inference FLOPs per sample': f"{inference_flops_per_sample:.2e}",
        'Space complexity': f"O(LSTM_units^2 + conv_filters * kernel_size * features)"
    }

complexity = estimate_complexity(best_model, X_train_main.shape, best_params['batch_size'], epochs_ran, best_params)

print("\n🧮 Big O Complexity Estimates:")
for k, v in complexity.items():
    print(f"{k}: {v}")

# Evaluation on Test Set
inference_start = time.time()
y_pred_scaled = best_model.predict(X_test, verbose=0)
inference_end = time.time()
inference_time_per_sample = (inference_end - inference_start) / len(X_test) if len(X_test) > 0 else np.nan

y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Standard metrics
mae = mean_absolute_error(y_test_original, y_pred)
mape = mean_absolute_percentage_error(y_test_original, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)
mbe = mean_bias_error(y_test_original, y_pred)

# Relative error based metrics
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

print('\n Best Model Test Performance:')
print(f"MAE: {mae:.4f} | nMAE: {nMAE:.6f}")
print(f"MAPE: {mape:.2f}% | RMSE: {rmse:.4f} | nRMSE: {nRMSE:.6f}")
print(f"MBE: {mbe:.4f} | nMBE: {nMBE:.6f}")
print(f"R²: {r2:.4f}")
print(f"MARE: {mare:.6f} | MSRE: {msre:.6f}")
print(f"RMSRE: {rmsre:.6f} | RMSPE: {rmspe:.2f}%")
print(f"sMAPE: {smape:.2f}% | MASE: {mase:.4f}")
print(f"Inference Time per Sample: {inference_time_per_sample*1000:.4f} ms")

# Predict the Next-Day
next_day_data = pd.read_excel('Input_to_be_predicted.xlsx')
for i in range(1, look_back + 1):
    next_day_data[f'power_lag{i}'] = next_day_data['power'].shift(i)
# Drop rows where any lag is missing (i.e., the first look_back rows)
next_day_data.dropna(subset=[f'power_lag{i}' for i in range(1, look_back + 1)], inplace=True)


timestamps = next_day_data['Timestamp']
X_next_day = next_day_data[feature_columns].values
y_next_day_actual = next_day_data['power'].values

X_next_day_scaled = scaler_X.transform(X_next_day)
X_next_day_scaled = X_next_day_scaled.reshape(X_next_day_scaled.shape[0], look_back, X_next_day_scaled.shape[1])

next_day_pred_start = time.time()
y_next_day_forecast_scaled = best_model.predict(X_next_day_scaled)
next_day_pred_end = time.time()
next_day_prediction_time = next_day_pred_end - next_day_pred_start

y_next_day_forecast = scaler_y.inverse_transform(y_next_day_forecast_scaled).flatten()

df_forecast = pd.DataFrame({
    'Time': timestamps,
    'Actual_Power': y_next_day_actual,
    'Forecasted_Power': y_next_day_forecast
})

# Metirces for the Next-day metrics
mae_next_day = mean_absolute_error(y_next_day_actual, y_next_day_forecast)
mape_next_day = mean_absolute_percentage_error(y_next_day_actual, y_next_day_forecast)
rmse_next_day = np.sqrt(mean_squared_error(y_next_day_actual, y_next_day_forecast))
r2_next_day = r2_score(y_next_day_actual, y_next_day_forecast)
mbe_next_day = mean_bias_error(y_next_day_actual, y_next_day_forecast)

rel_error_next = (y_next_day_actual - y_next_day_forecast) / (y_next_day_actual + eps)
mare_next = np.mean(np.abs(rel_error_next))
msre_next = np.mean(rel_error_next ** 2)
rmsre_next = np.sqrt(msre_next)
rmspe_next = rmsre_next * 100

smape_next = symmetric_mean_absolute_percentage_error(y_next_day_actual, y_next_day_forecast)
mase_next = mean_absolute_scaled_error(y_next_day_actual, y_next_day_forecast, mase_scaling_factor)

nMAE_next_day = mae_next_day / INSTALLED_CAPACITY
nRMSE_next_day = rmse_next_day / INSTALLED_CAPACITY
nMBE_next_day = mbe_next_day / INSTALLED_CAPACITY

total_time = time.time() - total_start_time

print('\n Best Model Next-Day Forecast Performance-random:')
print(f"MAE: {mae_next_day:.4f} | nMAE: {nMAE_next_day:.6f}")
print(f"MAPE: {mape_next_day:.2f}% | RMSE: {rmse_next_day:.4f} | nRMSE: {nRMSE_next_day:.6f}")
print(f"MBE: {mbe_next_day:.4f} | nMBE: {nMBE_next_day:.6f}")
print(f"R²: {r2_next_day:.4f}")
print(f"MARE: {mare_next:.6f} | MSRE: {msre_next:.6f}")
print(f"RMSRE: {rmsre_next:.6f} | RMSPE: {rmspe_next:.2f}%")
print(f"sMAPE: {smape_next:.2f}% | MASE: {mase_next:.4f}")
print(f"Next-Day Prediction Time: {next_day_prediction_time:.4f} s")
print(f"Total Script Execution Time: {total_time:.2f} s")

# Export results into excel
model_info = pd.DataFrame({
    'Model': ['CNN-LSTM'],
    'Trainable Params': [trainable_params],
    'Total Params': [total_params],
    'Batch Size': [best_params['batch_size']],
    'Look-back Window': [look_back],
    'Optimizer': ['Adam'],
    'Learning Rate': [best_params['learning_rate']],
    'Epochs Ran': [epochs_ran],
    'Actual_Memory Increase (MB)': [memory_increase_mb]
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

with pd.ExcelWriter('Turbine-CNN_LSTM_randomsearch_best_model.xlsx') as writer:
    model_info.to_excel(writer, sheet_name='Model_Info', index=False)
    timing_summary.to_excel(writer, sheet_name='Timing_Summary', index=False)
    test_metrics.to_excel(writer, sheet_name='Test_Performance', index=False)
    nextday_metrics.to_excel(writer, sheet_name='NextDay_Performance', index=False)
    df_forecast.to_excel(writer, sheet_name='NextDay_Forecast', index=False)
    complexity_df.to_excel(writer, sheet_name='BigO_Estimates', index=False)

print("\nAll metrics + forecasts + Big O exported to 'CNN_LSTM_randomsearch_best_model.xlsx'")

#  Plot
plt.figure(figsize=(12, 6))
plt.plot(df_forecast['Time'], df_forecast['Actual_Power'], label='Actual Power', marker='o', markersize=6, linewidth=2, color='blue')
plt.plot(df_forecast['Time'], df_forecast['Forecasted_Power'], label='Forecasted Power', marker='x', markersize=6, linewidth=2, color='red')
plt.xlabel('Time', fontsize=14, fontweight='bold')
plt.ylabel('Power Output (kW)', fontsize=14, fontweight='bold')
plt.title('Next Day Forecast vs Actual (CNN-LSTM)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=1.5, alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("CNN_LSTM_forecast_plot_by_random.pdf", dpi=500)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history_best.history['loss'], label='Training Loss')
plt.plot(history_best.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss - (CNN-LSTM)')
plt.legend()
plt.grid(True)
plt.savefig("CNN_LSTM_training_loss-Random-Search.pdf")
plt.show()

# XAI via SHAP
print("Starting SHAP analysis...")

background_size = min(50, X_train_main.shape[0])
background_indices = np.random.choice(X_train_main.shape[0], background_size, replace=False)
background_data = X_train_main[background_indices].astype(np.float32)

test_sample_size = min(50, X_test.shape[0])
test_indices = np.random.choice(X_test.shape[0], test_sample_size, replace=False)
X_test_sample = X_test[test_indices].astype(np.float32)

# Try GradientExplainer first
try:
    explainer = shap.GradientExplainer(best_model, background_data)
    shap_values = explainer.shap_values(X_test_sample)
    expected_value = explainer.expected_value
    print("Used GradientExplainer for SHAP.")
except Exception as e:
    print(f"GradientExplainer failed: {e}. Trying DeepExplainer...")
    try:
        # Fallback to DeepExplainer with explicit tensor
        background_tensor = tf.convert_to_tensor(background_data, dtype=tf.float32)
        explainer = shap.DeepExplainer(best_model, background_tensor)
        shap_values = explainer.shap_values(X_test_sample)
        expected_value = explainer.expected_value
        print("Used DeepExplainer for SHAP.")
    except Exception as e2:
        print(f"DeepExplainer also failed: {e2}. Falling back to KernelExplainer.")
        # Prepare flat data for KernelExplainer
        background_flat = background_data.reshape(background_data.shape[0], -1)
        X_test_flat = X_test_sample.reshape(X_test_sample.shape[0], -1)
        def predict_fn(data_2d):
            data_3d = data_2d.reshape((-1,) + X_train_main.shape[1:])
            preds = best_model.predict(data_3d, verbose=0)
            # If binary classification, extract positive class probability
            if preds.shape[-1] == 2:
                return preds[:, 1]
            else:
                return preds.flatten()
        explainer = shap.KernelExplainer(predict_fn, background_flat)
        shap_values = explainer.shap_values(X_test_flat, nsamples=200)
        expected_value = explainer.expected_value

# Process shap_values (assuming single output)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
if isinstance(expected_value, list):
    expected_value = expected_value[0]

# Continue with your plotting...

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_flat, feature_names=feature_columns, plot_type='bar', show=False)
    plt.title('Turbine-Feature Importance by SHAP - (CNN-LSTM)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Turbine-shap_feature_importance-cnn-lstm.pdf')
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_flat, feature_names=feature_columns, show=False)
    plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Turbine-shap_detailed_plot-cnn-lstm.pdf')
    plt.show()

    if isinstance(expected_value, np.ndarray):
        ev = float(expected_value) if expected_value.size == 1 else expected_value[0]
    else:
        ev = float(expected_value)

    try:
        fp = shap.force_plot(ev, shap_values[0], X_test_flat[0], feature_names=feature_columns, matplotlib=False)
        shap.save_html("shap_force_plot_sample0.html", fp)
        print("Saved interactive force plot to shap_force_plot_sample0.html")
    except Exception as fp_e:
        print(f"Interactive force plot failed ({fp_e}), trying matplotlib version.")
        try:
            shap.force_plot(ev, shap_values[0], X_test_flat[0], feature_names=feature_columns, matplotlib=True)
            plt.tight_layout()
            plt.savefig('shap_force_plot_sample0.png')
            plt.show()
            print("Saved matplotlib force plot to shap_force_plot_sample0.pdf")
        except Exception as plt_e:
            print(f"Matplotlib force plot also failed: {plt_e}")

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.flatten()
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance based on SHAP values:")
    print(feature_importance)
except Exception as e:
    print(f"Error in SHAP calculation: {e}")
    print("Continue without SHAP visualization...")

print("\nAll visualizations saved.")