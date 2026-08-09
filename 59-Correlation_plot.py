import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data2.csv")

# Label / target column
label = "Result"

# Select numeric features
features = df.select_dtypes(include="number").columns.drop(label)

# Create subplot grid
n = len(features)
cols = 2
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))

# Make axes 1D
axes = axes.flatten()

for i, feature in enumerate(features):

    # Correlation
    r = df[feature].corr(df[label])

    # Scatter plot
    axes[i].scatter(
        df[feature],
        df[label],
        alpha=0.7
    )

    axes[i].set_title(
        f"{feature} vs {label}  (r = {r:.3f})"
    )

    axes[i].set_xlabel(feature)
    axes[i].set_ylabel(label)
    axes[i].grid(True, alpha=0.3)

# Hide empty subplots
for i in range(len(features), len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()