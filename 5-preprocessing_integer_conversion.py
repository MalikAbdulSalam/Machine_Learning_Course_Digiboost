
import pandas as pd



# Read CSV file
df = pd.read_csv("data.csv")
print(df)

# forcefully change datatype into interger
df["Age"] = df["Age"].astype(int)

print(df["Age"])

df["Math"]  = df["Math"].astype(int)
print(df["Math"])