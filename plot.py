import matplotlib.pyplot as plt
import pandas as pd

from training_context import monitor

# Convert data to a DataFrame
df = pd.DataFrame(monitor.data)
# Save to CSV
df.to_csv("layer_statistics.csv", index=False)
# Separate means and standard deviations for plotting
layer_names = df["layer"]
means = df["mean"]
stds = df["std"]


# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Plot means
ax.bar(layer_names, means, yerr=stds, capsize=5, alpha=0.7)
ax.set_xlabel("Layers")
ax.set_ylabel("Mean Value")
ax.set_title("Mean and Standard Deviation of Layer Weights and Biases")
ax.set_xticklabels(layer_names, rotation=45, ha="right")

# Show the plot
plt.tight_layout()
plt.savefig("layer_statistics_plot.png")
plt.show()