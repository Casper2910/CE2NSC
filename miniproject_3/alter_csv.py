import pandas as pd

# names to remove
names_to_remove = ["cuda-numba", 'njit', 'cupy']

# load csv
df = pd.read_csv("benchmarks_wrong.csv")

# filter out rows where 'name' is in the list
df = df[~df["name"].isin(names_to_remove)]

# save result
df.to_csv("no_cuda_cupy_or_njit.csv", index=False)