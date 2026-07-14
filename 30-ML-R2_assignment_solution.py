from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import time

# Read dataset
df = pd.read_csv("C:/Users/PC/Desktop/data.csv")

print(df)

# Input and Output
X = df[["Physics", "English", "Chemistry","Math"]]
y = df["Result"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

print("################ model is in training phase #############")
time.sleep(3)

# Predict on training data
y_pred = model.predict(X)

# Evaluation
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)

r2 = r2_score(y, y_pred)
print("R² Score:", r2)



print("###################### predicting on new data #################")
time.sleep(3)

# New student's data
new_student = pd.DataFrame({
    "Physics": [85],
    "English": [90],
    "Chemistry": [88],
    "Math": [87]
})

# Predict
prediction = model.predict(new_student)

print("Predicted Result (Raw):", prediction[0])
