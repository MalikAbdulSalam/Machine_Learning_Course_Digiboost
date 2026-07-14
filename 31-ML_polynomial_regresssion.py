import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time


# 1. Read CSV
df = pd.read_csv("salary_data.csv")

# 2. Prepare data
X = df[["YearsExperience"]]
y = df["Salary"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# 3. Create polynomial features (degree 3)
poly = PolynomialFeatures(degree=3)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

print(X_poly_train)

print("#####################")
print(X_poly_test)


print("############# training started ##########")
# 4. Train polynomial regression
model = LinearRegression()
model.fit(X_poly_train, y_train)
time.sleep(3)

# 5. Make predictions
y_pred = model.predict(X_poly_test)



# 6. Evaluation Matric
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Value of MSE", mse)
print("Value of RMSE", rmse)
print("Value of MAE", mae)
print("value of r square", r2)


print("############## predicted values ##########")
# pridicting on new value
new_exp = [[13]]
new_exp_poly = poly.transform(new_exp)
pred = model.predict(new_exp_poly)
print(pred[0])