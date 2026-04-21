import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

# Average metrics
summary = df.groupby("type").mean(numeric_only=True)

print(summary)

# Plot latency
plt.figure()
for t in df["type"].unique():
    subset = df[df["type"] == t]
    plt.plot(subset["latency"].values, label=t)

plt.title("Latency Comparison")
plt.legend()
plt.xlabel("Prompt Index")
plt.ylabel("Latency (s)")
plt.show()

# Plot RAM
plt.figure()
for t in df["type"].unique():
    subset = df[df["type"] == t]
    plt.plot(subset["ram"].values, label=t)

plt.title("Memory Usage Comparison")
plt.legend()
plt.xlabel("Prompt Index")
plt.ylabel("RAM (GB)")
plt.show()