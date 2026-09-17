import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("non_linear.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Visualize the non-linear relationship
plt.scatter(df['YearsExperience'], df['Salary'], color='blue', alpha=0.7)
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.title("Experience vs Salary (Non-linear Relationship)")
plt.grid(True, alpha=0.3)
plt.show()


X = df[["YearsExperience"]]  # Feature
y = df["Salary"]             # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create polynomial features
poly = PolynomialFeatures(degree=1)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)



print("Original features shape:", X_train.shape)
print("Polynomial features shape:", X_poly_train.shape)


# Train on polynomial features
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)


y_hat_poly = poly_model.predict(X_poly_test)


poly_mse = mean_squared_error(y_test, y_hat_poly)
poly_r2 = r2_score(y_test, y_hat_poly)

print(poly_mse)
print(poly_r2)