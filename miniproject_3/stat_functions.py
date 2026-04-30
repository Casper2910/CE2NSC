import pandas as pd
import matplotlib.pyplot as plt
import os

# ---- CONFIG ----
FILE_PATH = "benchmarks_home_pc.csv"
OUTPUT_DIR = "plots"

GROUP_COL = "name"     # e.g. algorithm / method
X_COL = "size"         # e.g. input size
VALUE_COL = "time"     # e.g. runtime

AGGS = ["min", "mean", "max"]  # can change to any pandas agg funcs

# ----------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(FILE_PATH, on_bad_lines="skip")

df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")

df = df[
    df[GROUP_COL].notna() &
    df[X_COL].notna() &
    df[VALUE_COL].notna()
]

# Aggregate dynamically
stats = (
    df.groupby([GROUP_COL, X_COL])[VALUE_COL]
    .agg(AGGS)
    .reset_index()
)

# Generic plotting function
def plot_metric(metric):
    plt.figure()

    for group in stats[GROUP_COL].unique():
        subset = stats[stats[GROUP_COL] == group]
        plt.plot(subset[X_COL], subset[metric], marker="o", label=group)

    plt.xlabel(X_COL)
    plt.ylabel(f"{metric} {VALUE_COL}")
    plt.title(f"{metric.capitalize()} {VALUE_COL}")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{VALUE_COL}_{metric}.png", dpi=300)
    plt.close()

# Generate all plots
for metric in AGGS:
    plot_metric(metric)

# ---- Scaling analysis (log-log) ----
plt.figure()

for group in stats[GROUP_COL].unique():
    subset = stats[stats[GROUP_COL] == group].sort_values(X_COL)

    plt.loglog(
        subset[X_COL],
        subset["mean"],
        marker="o",
        label=group
    )

plt.xlabel(X_COL)
plt.ylabel(f"mean {VALUE_COL}")
plt.title("Scaling Analysis (log-log)")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/scaling_loglog.png", dpi=300)
plt.close()