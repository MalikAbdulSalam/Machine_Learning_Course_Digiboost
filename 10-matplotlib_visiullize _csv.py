import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

gender_counts = df["Gender"].value_counts()

plt.figure(figsize=(6,6))
plt.plot(df["Gender"], 'o')

plt.title("Gender Distribution")
plt.show()