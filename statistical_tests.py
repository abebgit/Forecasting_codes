import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, norm


#  Load data

input_file = "Turbine_next_day_prediction.xlsx"
output_file = "statistical_comparison_results.xlsx"
df = pd.read_excel(input_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])

y_true = df["actual"].to_numpy()

models = {
    "M1": df["M1"].to_numpy(),  # baseline LSTM
    "M2": df["M2"].to_numpy(),  # W-LSTM
    "M3": df["M3"].to_numpy(), #CNN-LSTM
    "M4": df["M4"].to_numpy(), # W-CNN
    "M5": df["M5"].to_numpy(),  # XGBoost
}

baselines = ["M1", "M5"]


# Error functions

def abs_error(y, yhat):
    return np.abs(y - yhat)

def sq_error(y, yhat):
    return (y - yhat) ** 2


# Diebold-Marino test

def diebold_mariano(e_model, e_base):
    d = e_model - e_base
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    dm_stat = d_mean / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value, d_mean


# Bootstrap confidence interval

def bootstrap_ci_improvement(e_model, e_base,
                             n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(e_model)
    deltas = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        deltas[i] = np.mean(e_base[idx] - e_model[idx])

    mean_delta = np.mean(deltas)
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))

    return mean_delta, lo, hi


# Run tests and collect outputs

rows_t, rows_w, rows_dm, rows_ci = [], [], [], []

for base in baselines:
    e_base_mae = abs_error(y_true, models[base])
    e_base_rmse = sq_error(y_true, models[base])

    for name, y_pred in models.items():
        if name == base:
            continue

        e_mae = abs_error(y_true, y_pred)
        e_rmse = sq_error(y_true, y_pred)

        # Paired t-test
        t_mae, p_t_mae = ttest_rel(e_mae, e_base_mae)
        t_rmse, p_t_rmse = ttest_rel(e_rmse, e_base_rmse)

        rows_t.append([name, base, "MAE", t_mae, p_t_mae])
        rows_t.append([name, base, "RMSE", t_rmse, p_t_rmse])

        # Wilcoxon
        w_mae, p_w_mae = wilcoxon(e_mae, e_base_mae)
        w_rmse, p_w_rmse = wilcoxon(e_rmse, e_base_rmse)

        rows_w.append([name, base, "MAE", w_mae, p_w_mae])
        rows_w.append([name, base, "RMSE", w_rmse, p_w_rmse])

        # Diebold–Mariano
        dm_mae, p_dm_mae, d_mae = diebold_mariano(e_mae, e_base_mae)
        dm_rmse, p_dm_rmse, d_rmse = diebold_mariano(e_rmse, e_base_rmse)

        rows_dm.append([name, base, "MAE", dm_mae, p_dm_mae, d_mae])
        rows_dm.append([name, base, "RMSE", dm_rmse, p_dm_rmse, d_rmse])

        # Bootstrap CI
        mean_mae, lo_mae, hi_mae = bootstrap_ci_improvement(e_mae, e_base_mae)
        mean_rmse, lo_rmse, hi_rmse = bootstrap_ci_improvement(e_rmse, e_base_rmse)

        rows_ci.append([name, base, "MAE", mean_mae, lo_mae, hi_mae])
        rows_ci.append([name, base, "RMSE", mean_rmse, lo_rmse, hi_rmse])


# Dataframe creations

df_t = pd.DataFrame(rows_t,
    columns=["Model", "Baseline", "Metric", "t_statistic", "p_value"])

df_w = pd.DataFrame(rows_w,
    columns=["Model", "Baseline", "Metric", "wilcoxon_statistic", "p_value"])

df_dm = pd.DataFrame(rows_dm,
    columns=["Model", "Baseline", "Metric",
             "DM_statistic", "p_value", "Mean_Delta_Loss"])

df_ci = pd.DataFrame(rows_ci,
    columns=["Model", "Baseline", "Metric",
             "Mean_Improvement", "CI_Lower_95", "CI_Upper_95"])


# Export to excel

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_t.to_excel(writer, sheet_name="Paired_t_test", index=False)
    df_w.to_excel(writer, sheet_name="Wilcoxon_test", index=False)
    df_dm.to_excel(writer, sheet_name="Diebold_Mariano", index=False)
    df_ci.to_excel(writer, sheet_name="Bootstrap_CI", index=False)

print(f"All tests saved to: {output_file}")
