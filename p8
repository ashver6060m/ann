
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential()

model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy_sklearn = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy using sklearn: {accuracy_sklearn * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

sample_idx = 0
sample_input = X_test[sample_idx].reshape(1, -1)
sample_prediction = model.predict(sample_input)
predicted_class = np.argmax(sample_prediction)
class_labels = ['Setosa', 'Versicolor', 'Virginica']
print(f"Predicted class for sample {sample_idx}: {class_labels[predicted_class]}")

print(f"Probabilities for each class: {sample_prediction[0]}")

plt.figure(figsize=(8, 6))
plt.bar(class_labels, sample_prediction[0])
plt.title('Class Probabilities for Sample')
plt.xlabel('Class')
plt.ylabel('Probability')
plt.show()
