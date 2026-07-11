from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time

# Sample dataset
X = [[1], [2], [3], [4], [5]]   # Input (Hours Studied)
y = [40, 50, 60, 70, 80]         # Output (Marks)

# Create linear regression model
model = LinearRegression()

# Train model
model.fit(X, y)

print("################ model is in training phase #############")
time.sleep(3)


# Predict on training data
y_pred = model.predict(X)

# Calculate MSE
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)

print("###################### predicting phase #################")
time.sleep(3)

# Predict for new data
prediction = model.predict([[3.5]])

print("Predicted Marks:", prediction[0])