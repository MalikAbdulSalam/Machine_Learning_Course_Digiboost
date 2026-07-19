import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For feature importance
from sklearn.inspection import permutation_importance


df = pd.read_csv("house_prices.csv")
print(df)

print("#####################################")

# Define features (all except Price)
feature_columns = ['Size', 'Bedrooms', 'Bathrooms', 'Age']
X = df[feature_columns]
y = df['Price']

print("Features ")
print(X)
print("Labels")
print(y)


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train_scaled)
X_poly_test = poly.transform(X_test_scaled)

# Train polynomial model
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# 6. Predict
y_pred = poly_model.predict(X_poly_test)

# 7. Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("MSE: ", mse)
print("RMSE: ", rmse)
print("MAE: ", mae)
print("R2: ", r2)


# 9. Predict new house
new_house = [[2800, 4, 3, 2]]
new_house_scaled = scaler.transform(new_house)
new_house_poly = poly.transform(new_house_scaled)
pred = poly_model.predict(new_house_poly)
print(f"Predicted price: ${pred[0]:,.2f}")



# 8. Visualize - Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"Predicted vs Actual\nR² = {r2:.4f}, RMSE = ${rmse:,.0f}")
plt.grid(True, alpha=0.3)
plt.show()