import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("non_linear.csv")


X = df[["YearsExperience"]]  # Feature
y = df["Salary"]             # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




# Train on linear features
Linear_model = LinearRegression()
Linear_model.fit(X_train, y_train)

y_hat_Linear = Linear_model.predict(X_test)


Linear_mse = mean_squared_error(y_test, y_hat_Linear)
Linear_r2 = r2_score(y_test, y_hat_Linear)

print(Linear_mse)
print(Linear_r2)