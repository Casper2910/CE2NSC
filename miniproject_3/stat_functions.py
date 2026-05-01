import pandas as pd
import matplotlib.pyplot as plt
import os

# ---- CONFIG ----
FILE_PATH = "new_new_benchmarks.csv"
OUTPUT_DIR = "new_plots"

GROUP_COL = "name"
X_COL = "size"
VALUE_COL = "time"

AGGS = ["min", "mean", "max"]
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

# Aggregate including max_iter
stats = (
    df.groupby([GROUP_COL, X_COL, "max_iter"])[VALUE_COL]
    .agg(AGGS)
    .reset_index()
)

# ---- Plotting ----
def plot_metric(metric, max_iter):
    plt.figure()

    subset_all = stats[stats["max_iter"] == max_iter]

    for group in subset_all[GROUP_COL].unique():
        subset = subset_all[subset_all[GROUP_COL] == group]
        plt.plot(subset[X_COL], subset[metric], marker="o", label=group)

    plt.xlabel(X_COL)
    plt.ylabel(f"{metric} {VALUE_COL}")
    plt.title(f"{metric.capitalize()} {VALUE_COL} (max_iter={max_iter})")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/{VALUE_COL}_{metric}_iter_{max_iter}.png", dpi=300)
    plt.close()

# Generate plots per max_iter
for max_iter in stats["max_iter"].unique():
    for metric in AGGS:
        plot_metric(metric, max_iter)

# ---- Scaling analysis (log-log) ----
for max_iter in stats["max_iter"].unique():
    plt.figure()

    subset_all = stats[stats["max_iter"] == max_iter]

    for group in subset_all[GROUP_COL].unique():
        subset = subset_all[subset_all[GROUP_COL] == group].sort_values(X_COL)

        plt.loglog(
            subset[X_COL],
            subset["mean"],
            marker="o",
            label=group
        )

    plt.xlabel(X_COL)
    plt.ylabel(f"mean {VALUE_COL}")
    plt.title(f"Scaling Analysis (max_iter={max_iter})")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/scaling_loglog_iter_{max_iter}.png", dpi=300)
    plt.close()