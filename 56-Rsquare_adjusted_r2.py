# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data: house size (X) → price (Y)
X = np.array([500, 600, 700, 800, 900]).reshape(-1, 1)
y = np.array([100, 120, 150, 180, 200])


# Train model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)



# R²
r2 = r2_score(y, y_pred)



# Adjusted R² (manual)
n = len(y) # observations
k = X.shape[1] # predictors
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)




print(f"R² = {r2:.4f}")
print(f"Adjusted R² = {adj_r2:.4f}")