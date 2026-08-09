import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data2.csv")

# Correlation of every feature with the label
correlation = df.corr(numeric_only=True)["Result"].sort_values(ascending=False)

print(correlation)



# Draw correlations

# Load dataset
df = pd.read_csv("data2.csv")

# Label/target column
label = "Result"

# Calculate correlation with label
correlation = df.corr(numeric_only=True)[label].drop(label)

# Sort correlations
correlation = correlation.sort_values()

# Display values
print(correlation)

# Draw bar chart
plt.figure(figsize=(10, 6))
plt.bar(correlation.index, correlation.values)

plt.axhline(0, linewidth=1)
plt.xlabel("Features")
plt.ylabel("Correlation with Result")
plt.title("Correlation Between Features and Label")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()