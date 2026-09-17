import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



df = pd.read_csv("house_prices.csv")
print(df)


# Define features (all except Price)
feature_columns = ['Size', 'Bedrooms', 'Bathrooms', 'Age']
X = df[feature_columns]
y = df['Price']


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
print("Model Trained")


# make model and train it
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)



# Make predictions
y_hat_poly = poly_model.predict(X_poly_test)


# calculate matrix
poly_mse = mean_squared_error(y_test, y_hat_poly)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, y_hat_poly)

print(poly_mse)
print(poly_rmse)
print(poly_r2)



