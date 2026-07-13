import pandas as pd
import numpy as np

# ---------------- CONFIGURATION ----------------
INPUT_FILE = "Raw_data.xlsx"
OUTPUT_TRAIN = "training.xlsx"
OUTPUT_TEST = "testing.xlsx"
OUTPUT_COMPILED = "preproccessed_data.xlsx"
FREQ = "5T"
TRAIN_YEARS = [2019, 2020, 2021, 2022]
TEST_YEAR = 2023


GAP_THRESHOLD_STEPS = 12   # 1h / 5min = 12

value_cols = ['Wind_Speed', 'Pitch_Angle', 'Power', 'Outdoor_Temperature']


# ============================================================
# UTILITIES
# ============================================================
def fill_past_only(series):
    #fill-forward, and fill-backward only at the beginning
    series = series.sort_index()
    return series.ffill().bfill()
#outlier detection via MAD
def detect_outliers_past(series, window=36, k=3):
    series = series.sort_index()
    med = series.rolling(window, min_periods=1).median()
    mad = (series - med).abs().rolling(window, min_periods=1).median()
    mad = mad.replace(0, np.nan).ffill().bfill()
    lo = med - k * mad
    hi = med + k * mad
    return (series < lo) | (series > hi)

# Load raw SCADA data

df_raw = pd.read_excel(INPUT_FILE)
df_raw.columns = df_raw.columns.str.replace(" ", "_")
df_raw['Time'] = pd.to_datetime(df_raw['Time'])
df_raw = df_raw.set_index("Time").sort_index()

print("Loaded raw rows:", len(df_raw))


# ============================================================
# Strictly build 5-min grid chronologically and ensert all missing timestamps
# ============================================================
full_index = pd.date_range(df_raw.index.min(),
                           df_raw.index.max(),
                           freq=FREQ)

df = df_raw.reindex(full_index)
print("Rows after inserting ALL missing timestamps:", len(df))


# Identify long and short gaps before imputation

# missing_mask must be pd.Series (fixes your AttributeError)
missing_mask = pd.Series(~full_index.isin(df_raw.index),
                         index=full_index)

groups = (missing_mask != missing_mask.shift()).cumsum()
gap_sizes = missing_mask.groupby(groups).sum()

short_gap_ids = gap_sizes[(gap_sizes > 0) & (gap_sizes <= GAP_THRESHOLD_STEPS)].index
long_gap_ids  = gap_sizes[gap_sizes > GAP_THRESHOLD_STEPS].index

short_gap_ts = []
long_gap_ts  = []

for gid in gap_sizes.index:
    ts_block = full_index[groups == gid]
    if gid in short_gap_ids:
        short_gap_ts.extend(ts_block)
    elif gid in long_gap_ids:
        long_gap_ts.extend(ts_block)

short_gap_ts = pd.DatetimeIndex(short_gap_ts)
long_gap_ts  = pd.DatetimeIndex(long_gap_ts)

print("Short-gap timestamps:", len(short_gap_ts))
print("Long-gap timestamps :", len(long_gap_ts))


# Imputation by filling only short gaps

for col in value_cols:
    filled = fill_past_only(df[col])
    filled[long_gap_ts] = np.nan   # keep long gaps NaN
    df[col] = filled

# Outlier removal + refill short gaps only
for col in value_cols:
    mask = detect_outliers_past(df[col])
    mask = mask & (~df.index.isin(long_gap_ts))
    df.loc[mask, col] = np.nan

    filled = fill_past_only(df[col])
    filled[long_gap_ts] = np.nan
    df[col] = filled

# Physics constraints(positve values for power and speed)
df["Power"] = df["Power"].clip(lower=0)
df["Wind_Speed"] = df["Wind_Speed"].clip(lower=0)

# Feature engineering
df["WindSpeed_change_per_min"] = df["Wind_Speed"].diff() / 5
df["WindSpeed_mean_30min"]     = df["Wind_Speed"].rolling(6).mean().shift(1)
df["Power_mean_30min"]         = df["Power"].rolling(6).mean().shift(1)

df["hour"]      = df.index.hour
df["dayofyear"] = df.index.dayofyear
df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)

# filling continue for short gaps
df.loc[~df.index.isin(long_gap_ts)] = df.loc[~df.index.isin(long_gap_ts)].ffill()


# Remove long gaps at this stage

df = df[~df.index.isin(long_gap_ts)]

print("Rows after long-gap removal:", len(df))


# Split training and test data chronologically

df["year"] = df.index.year
train_df = df[df["year"].isin(TRAIN_YEARS)]
test_df  = df[df["year"] == TEST_YEAR]

print("Train rows:", len(train_df))
print("Test rows: ", len(test_df))


# extend six records unfilled from the SCADA data in 2023
# ============================================================
last_ts = test_df.index.max()
target_ts = last_ts.normalize() + pd.Timedelta("23:55:00")

if last_ts < target_ts:
    extra_ts = pd.date_range(last_ts + pd.Timedelta("5min"),
                             target_ts,
                             freq="5T")
    test_df = test_df.reindex(test_df.index.union(extra_ts))

# Fill core SCADA for extended rows
test_df[value_cols] = test_df[value_cols].ffill()


# Compute feature engineering on the last extended records

test_df["WindSpeed_change_per_min"] = test_df["Wind_Speed"].diff() / 5
test_df["WindSpeed_mean_30min"]     = test_df["Wind_Speed"].rolling(6).mean().shift(1)
test_df["Power_mean_30min"]         = test_df["Power"].rolling(6).mean().shift(1)

test_df["hour"]      = test_df.index.hour
test_df["dayofyear"] = test_df.index.dayofyear
test_df["hour_sin"]  = np.sin(2*np.pi*test_df["hour"]/24)
test_df["hour_cos"]  = np.cos(2*np.pi*test_df["hour"]/24)

# Final forward-fill to ensure last rows have No NaNs
test_df = test_df.ffill()


# Save outputs

train_df = train_df.reset_index().rename(columns={"index":"Time"})
test_df  = test_df.reset_index().rename(columns={"index":"Time"})

train_df.to_excel(OUTPUT_TRAIN, index=False)
test_df.to_excel(OUTPUT_TEST, index=False)

compiled = pd.concat([train_df, test_df], ignore_index=True)
compiled = compiled.sort_values("Time")
compiled.to_excel(OUTPUT_COMPILED, index=False)

print("\nPreprocessing Completed.")
