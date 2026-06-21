import pandas as pd
# to read our csv file
df = pd.read_csv("data.csv")
print(df)

basket = df["Result"]
print(basket)

print("#####################################################")

basket2 = df["Result"]
print(basket2)