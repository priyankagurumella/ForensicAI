import cv2
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

real_path = "deepfake_dataset/training_real"
fake_path = "deepfake_dataset/training_fake"

X = []
Y = []

# Real Images
for image_name in os.listdir(real_path)[:100]:

    image_path = os.path.join(real_path, image_name)

    img = cv2.imread(image_path)

    img = cv2.resize(img, (128, 128))

    X.append(img)

    Y.append(0)

# Fake Images
for image_name in os.listdir(fake_path)[:100]:

    image_path = os.path.join(fake_path, image_name)

    img = cv2.imread(image_path)

    img = cv2.resize(img, (128, 128))

    X.append(img)

    Y.append(1)

X = np.array(X)

Y = np.array(Y)

# Convert images to 1D vectors
X = X.reshape(len(X), -1)

print("Dataset Shape:", X.shape)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42
)

model.fit(X_train, Y_train)

# Save model
with open("models/deepfake_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Deepfake Model Saved Successfully")

# Prediction
Y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print()
print("================================")
print("DEEPFAKE DETECTION MODEL")
print("================================")
print()

print("Model Accuracy:", round(accuracy * 100, 2), "%")