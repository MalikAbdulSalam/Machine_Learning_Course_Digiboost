import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


data = {
    "Height":[160,165,170,175,180,185],
    "Weight":[55,60,65,70,75,80],
    "Class":[0,0,0,1,1,1]
}
print("################# this is dictionary data ###############")
print(data)

df = pd.DataFrame(data)

print("################# this is panda dataframe ###############")
print(df)




X = df[["Height","Weight"]]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)


#predictin to calculate evaluation matric
y_pred = knn.predict(X_test)

A_score = accuracy_score(y_test, y_pred)
print(A_score)




# -----------------------------
# Predict on New Data
# -----------------------------

# New person's Height and Weight
new_data = [[172, 68]]

# Apply the same scaling used for training
new_data_scaled = scaler.transform(new_data)

# Predict the class
prediction = knn.predict(new_data_scaled)

print("Predicted Class:", prediction[0])

if prediction[0] == 0:
    print("Class 0")
else:
    print("Class 1")