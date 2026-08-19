# Pandas data ko read aur handle karne ke liye
import pandas as pd

# Dataset ko train aur test mein divide karne ke liye
from sklearn.model_selection import train_test_split

# Decision Tree Classification model
from sklearn.tree import DecisionTreeClassifier

# Model ki accuracy check karne ke liye
from sklearn.metrics import accuracy_score

# Confusion Matrix import karna
from sklearn.metrics import confusion_matrix

# Classification Report import karna
from sklearn.metrics import classification_report

# Decision Tree ka graph banane ke liye
from sklearn.tree import plot_tree

# Graph show karne ke liye
import matplotlib.pyplot as plt

#confiuion matrix check karna ka lia
from sklearn.metrics import confusion_matrix

# =====================================================
# STEP 1: Titanic Dataset Read
# =====================================================

# Titanic dataset ko read karna
df = pd.read_csv("Titanic-Dataset.csv")

# Dataset ki first 5 rows dekhna
print(df.head())


# =====================================================
# STEP 2: Dataset Information
# =====================================================

# Dataset ke columns aur data types check karna
print(df.info())


# =====================================================
# STEP 3: Missing Values Check
# =====================================================

# Har column mein missing values check karna
print(df.isnull().sum())


# =====================================================
# STEP 4: Missing Values Handle
# =====================================================

# Age ki missing values ko median se fill karna
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked ki missing values ko mode se fill karna
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin column ko remove karna
df = df.drop("Cabin", axis=1)

# Dobara missing values check karna
print(df.isnull().sum())


# =====================================================
# STEP 5: Unnecessary Columns Remove
# =====================================================

# Unnecessary columns remove karna
df = df.drop(["PassengerId", "Name", "Ticket"], axis=1)

# Dataset check karna
print(df.head())


# =====================================================
# STEP 6: Text ko Numbers mein Convert Karna
# =====================================================

# Sex column:
# male = 0
# female = 1

df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})


# Embarked column:
# S = 0
# C = 1
# Q = 2

df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

# Dataset check karna
print(df.head())


# =====================================================
# STEP 7: Features (X) aur Target (y)
# =====================================================

# X mein input/features rakhenge
X = df.drop("Survived", axis=1)

# y mein target/output rakhenge
y = df["Survived"]

# X check karna
print("Features:")
print(X.head())

# y check karna
print("Target:")
print(y.head())


# =====================================================
# STEP 8: Train Test Split
# =====================================================

# Dataset ko training aur testing data mein divide karna
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Training data ka size check karna
print("Training data:", X_train.shape)

# Testing data ka size check karna
print("Testing data:", X_test.shape)


# =====================================================
# STEP 9: Decision Tree Model
# =====================================================

# Decision Tree model banana
# max_depth=3 se tree sirf 3 levels tak jayega

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

# Model ko training data par train karna
model.fit(X_train, y_train)

# Message show karna
print("Decision Tree Model Training Completed")


# =====================================================
# STEP 10: Prediction
# =====================================================

# Test data par prediction karna
y_pred = model.predict(X_test)

# Predictions show karna
print("Predictions:")
print(y_pred)


# =====================================================
# STEP 11: Accuracy
# =====================================================

# Actual result aur predicted result ko compare karna
accuracy = accuracy_score(y_test, y_pred)

# Accuracy print karna
print("Model Accuracy:", accuracy)

# Accuracy percentage mein
print("Accuracy Percentage:", accuracy * 100, "%")


# =====================================================
# STEP 12: Confusion Matrix
# =====================================================

# Confusion Matrix calculate karna
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix print karna
print("Confusion Matrix:")
print(cm)


# =====================================================
# STEP 13: Classification Report
# =====================================================

# Classification Report generate karna
report = classification_report(y_test, y_pred)

# Report show karna
print("Classification Report:")
print(report)


# =====================================================
# STEP 14: Decision Tree Graph
# =====================================================

# Graph ka size
plt.figure(figsize=(15, 8))

# Decision Tree ko draw karna
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True
)

# Graph ka title
plt.title("Titanic Decision Tree Classification")

# Graph show karna
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Graph
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])

plt.show()

#Agar values bhi print karni hain
TN, FP, FN, TP = cm.ravel()

print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Positive:", TP)

print("Confusion Matrix:")
print(cm)
#is ka graph further step by step

labels = ["TN", "FP", "FN", "TP"]
values = [TN, FP, FN, TP]

plt.figure(figsize=(8, 5))
plt.bar(labels, values)

plt.xlabel("Confusion Matrix Classes")
plt.ylabel("Number of Predictions")
plt.title("Confusion Matrix - TN, FP, FN, TP")

plt.show()
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])

# Boxes ke andar TN, FP, FN, TP
plt.text(0, 0, f"TN\n{TN}", ha="center", va="center", fontsize=16)
plt.text(1, 0, f"FP\n{FP}", ha="center", va="center", fontsize=16)
plt.text(0, 1, f"FN\n{FN}", ha="center", va="center", fontsize=16)
plt.text(1, 1, f"TP\n{TP}", ha="center", va="center", fontsize=16)

plt.colorbar()

plt.show()
