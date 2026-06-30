import pandas as pd
from sklearn.model_selection import train_test_split

# Read CSV file
df = pd.read_csv("data.csv")
print(df)



print("############################################################")

# X are features
# Y are labels

X = df[["StudentID", "Name" , "Age" , "Gender" , "Math" ,  "Physics", "Chemistry", "English", "Attendance"]]
Y = df["Result"]

print(X)
print(Y)


print("################################# spliting data into 4 chunks ############################")

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("Triaining data features")
print(X_train)

print("Testing Data Features")
print(X_test)

print("labels of train dataset")
print(y_train)

print("labels of test dataset")
print(y_test )