from sklearn.linear_model import LinearRegression
import time

# Sample dataset
X = [[1], [2], [3], [4], [5]]   # Input (Hours Studied)
y = [40, 50, 60, 70, 80]


# create linear regression model

model = LinearRegression()


#train model
model.fit(X, y)

print("################ model is in training phase #############")
time.sleep(3)


print("###################### predicting phase #################") 
# predict model
prediction = model.predict([[8]])
time.sleep(3)

print("Predicted Marks:", prediction[0])


[[6]]