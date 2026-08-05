import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data2.csv")
print(df)



# seprate features and labels
y = df['Result']



plt.boxplot(y)

plt.show()

plt.scatter(df.index,y)
plt.show()


plt.hist(y)
plt.show()
