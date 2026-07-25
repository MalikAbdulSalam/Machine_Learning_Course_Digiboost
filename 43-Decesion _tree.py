import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("data2.csv")
print(df.head())



# seprate features and labels
X = df[['Age', 'English']]
y = df['Result']


# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
) 



# create a model 
model = DecisionTreeClassifier(random_state=42)


# Train the model
model.fit(X_train, y_train)


# prediction onf test dataset
y_pred = model.predict(X_test)
print(y_pred)



print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True
)

plt.show()