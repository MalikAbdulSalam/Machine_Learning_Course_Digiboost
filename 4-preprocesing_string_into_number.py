
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Read CSV file
df = pd.read_csv("data.csv")
print(df)

print("###################################")

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])

print(df)

df["Result"] = le.fit_transform(df["Result"])
print(df)