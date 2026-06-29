import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read CSV file
df = pd.read_csv("data.csv")
print(df)

# Create scaler
scaler = MinMaxScaler(feature_range=(-100, 100))

# Scale the Age column
df["Age"] = scaler.fit_transform(df[["Age"]])

print(df["Age"])


# Scale the Age column
df[["Age", "Math"]] = scaler.fit_transform(df[["Age", "Math"]])

print(df[["Age", "Math"]])

