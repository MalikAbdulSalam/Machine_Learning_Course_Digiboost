from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import calinski_harabasz_score



# create dataset
data = {

    "Age": [
        20, 22, 25, 27, 30,
        45, 48, 50, 52, 55
    ],

    "Income": [
        20000,
        22000,
        25000,
        27000,
        30000,
        60000,
        65000,
        70000,
        72000,
        75000
    ],

    "SpendingScore": [
        80,
        75,
        85,
        70,
        78,
        40,
        35,
        30,
        25,
        20
    ]
}

df = pd.DataFrame(data)



# mark data features

X = df[
    [
        "Age",
        "Income",
        "SpendingScore"
    ]
]

print(X)



# apply minmax scalar

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled)





# elbow method to find number of k

inertias = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(
        model.inertia_
    )


plt.plot(
    range(2, 11),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.show()




#evalutation matrics



score = calinski_harabasz_score(
    X_scaled,
    labels
)

print(
    "Calinski-Harabasz Score:",
    score
)