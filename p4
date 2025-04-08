import numpy as np
from sklearn.metrics import accuracy_score

data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
X = data[:, 0:2]
y = data[:, 2]

centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
k = len(centers)

weights = np.random.randn(k)
learning_rate = 0.1
epochs = 100

def rbf(x, center):
    return np.exp(-np.sum((x - center) ** 2) / 2)

def compute_rbf_matrix(X, centers):
    rbf_matrix = []
    for i in range(len(X)):
        rbf_row = []
        for j in range(len(centers)):
            rbf_row.append(rbf(X[i], centers[j]))
        rbf_matrix.append(rbf_row)
    return rbf_matrix  

for epoch in range(epochs):
    rbf_matrix = compute_rbf_matrix(X, centers)  
    outputs = []
    for i in range(len(X)):
        output = 0
        for j in range(k):
            output += rbf_matrix[i][j] * weights[j]
        outputs.append(output)

    errors = [y[i] - outputs[i] for i in range(len(y))]

    for j in range(k):  
        gradient = 0
        for i in range(len(X)): 
            gradient += errors[i] * rbf_matrix[i][j]
        weights[j] += learning_rate * gradient

def predict(X, centers, weights):
    pred = []
    for i in range(len(X)):
        output = 0
        for j in range(len(centers)):
            output += rbf(X[i], centers[j]) * weights[j]
        pred.append(1 if output >= 0.5 else 0)
    return pred

pred = predict(X, centers, weights)
print("Predicted:", pred)
print("Accuracy:", accuracy_score(y, pred) * 100, "%")
