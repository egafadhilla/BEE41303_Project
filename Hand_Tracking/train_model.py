import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# Dataset filename
csv_file = 'hand_dataset.csv'

if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' not found!")
    print("Please run 'collect_data.py' first to collect the data.")
    exit()

print("Loading dataset...")
data = pd.read_csv(csv_file, header=None)

# Separate features (X) and labels (y)
# Column 0 is the label, the rest are x, y, z coordinates
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

print(f"Total data: {len(data)}")

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nCurrently training model (Support Vector Machine)...")
# Initialize SVM model (very suitable for lightweight devices like Jetson Nano)
model = SVC(kernel='linear', probability=True)

# Train the model with training data
model.fit(X_train, y_train)

# Make predictions on testing data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"-> Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model_filename = 'hand_model.pkl'
joblib.dump(model, model_filename)
print(f"Model successfully saved to '{model_filename}'.")
print("Now you can run 'main.py' to detect gestures.")