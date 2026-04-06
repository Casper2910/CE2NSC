import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure output folder exists
os.makedirs("plots", exist_ok=True)

# Load data
df = pd.read_csv("benchmarks.csv")

# Aggregate stats
stats = (
    df.groupby(["method", "size"])["time"]
    .agg(["mean", "min", "max"])
    .reset_index()
)

# ---- Plot MIN ----
plt.figure()
for method in stats["method"].unique():
    subset = stats[stats["method"] == method]
    plt.plot(subset["size"], subset["min"], marker="o", label=method)

plt.xlabel("Size")
plt.ylabel("Min Time")
plt.title("Min Runtime")
plt.legend()
plt.savefig("plots/performance_min.png", dpi=300)
plt.close()

# ---- Plot MEAN ----
plt.figure()
for method in stats["method"].unique():
    subset = stats[stats["method"] == method]
    plt.plot(subset["size"], subset["mean"], marker="o", label=method)

plt.xlabel("Size")
plt.ylabel("Mean Time")
plt.title("Mean Runtime")
plt.legend()
plt.savefig("plots/performance_mean.png", dpi=300)
plt.close()

# ---- Plot MAX ----
plt.figure()
for method in stats["method"].unique():
    subset = stats[stats["method"] == method]
    plt.plot(subset["size"], subset["max"], marker="o", label=method)

plt.xlabel("Size")
plt.ylabel("Max Time")
plt.title("Max Runtime")
plt.legend()
plt.savefig("plots/performance_max.png", dpi=300)
plt.close()