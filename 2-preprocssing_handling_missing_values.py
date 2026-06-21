
import pandas as pd

# Read CSV file
df = pd.read_csv("data.csv")
print(df)

print("##############################")
#
# # Replace missing values in Age with mean Age
# df["Age"] = df["Age"].fillna(df["Age"].mean())
#
# print(df)

print("##############################")

# Replace missing values with mode
# df["Age"] = df["Age"].fillna(df["Age"].mode()[0])
#
# print(df)

print("###############################")


# Replace missing values with median
df["Age"] = df["Age"].fillna(df["Age"].median())

print(df)

# print the age column only
print(df["Age"])