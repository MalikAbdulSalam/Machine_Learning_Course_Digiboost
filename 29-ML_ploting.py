from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import time

# ============================
# Sample Dataset
# ============================
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([40, 50, 60, 70, 80])

# ============================
# Create Model
# ============================
model = LinearRegression()

print("################ Training Model ################")
time.sleep(2)

# Train
model.fit(X, y)

# ============================
# Predictions
# ============================
y_pred = model.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

print("\n################ Prediction ################")
time.sleep(2)

new_x = np.array([[3.5]])
prediction = model.predict(new_x)

print("Hours Studied :", new_x[0][0])
print("Predicted Marks :", prediction[0])

# ============================
# Regression Equation
# ============================
slope = model.coef_[0]
intercept = model.intercept_

print("\nRegression Equation")
print(f"Marks = {slope:.2f} × Hours + {intercept:.2f}")

# ============================
# Plot
# ============================
plt.figure(figsize=(8,6))

# Original data
plt.scatter(
    X,
    y,
    color='blue',
    s=100,
    label='Actual Data'
)

# Regression line
plt.plot(
    X,
    y_pred,
    color='red',
    linewidth=3,
    label='Regression Line'
)

# Predicted point
plt.scatter(
    new_x,
    prediction,
    color='green',
    s=150,
    marker='*',
    label='Prediction (3.5 hrs)'
)

# Dashed helper lines
plt.plot(
    [3.5, 3.5],
    [0, prediction[0]],
    linestyle='--',
    color='green'
)

plt.plot(
    [0, 3.5],
    [prediction[0], prediction[0]],
    linestyle='--',
    color='green'
)

# Labels
plt.title("Simple Linear Regression", fontsize=16)
plt.xlabel("Hours Studied", fontsize=12)
plt.ylabel("Marks", fontsize=12)

# Equation + Metrics
text = (
    f"Equation: y = {slope:.2f}x + {intercept:.2f}\n"
    f"MSE = {mse:.2f}\n"
    f"R² = {r2:.2f}"
)

plt.text(
    1.1,
    73,
    text,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8)
)

plt.grid(True)
plt.legend()

plt.show()