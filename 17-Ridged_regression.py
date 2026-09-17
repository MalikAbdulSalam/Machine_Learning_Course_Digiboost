import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


# read data file
df = pd.read_csv("student_marks.csv")


# Data split into 2 part
X = df[["Hours"]]   # Feature (independent)
y = df["Marks"]     # Target (dependent)



# data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% for testing
    random_state=42     # reproducible split
)



# create model
model = Ridge(alpha=1.0)


# train model
model.fit(X_train, y_train)


# predictions
y_hat = model.predict(X_test)



# Calculate metrics
mse = mean_squared_error(y_test, y_hat)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_hat)
r2 = r2_score(y_test, y_hat)



print("MSE is :    ", mse)
print("RMSE is :    ", rmse)
print("MAE is :    ", mae)
print("R2 is :    ", r2)




