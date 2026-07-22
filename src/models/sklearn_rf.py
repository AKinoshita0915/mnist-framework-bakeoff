# Import required packages
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from src.data_loader import get_sklearn_data

# Load Data
(X_train, y_train), (X_test, y_test) = get_sklearn_data()

# Model Architecture
# 100 trees and use all available cores for parallel processing
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model
print("Starting training...") # Visualize user what is happening
model.fit(X_train, y_train)

# Evaluation
print("Evaluating model...") # Visualize user what is happening
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%\n")

# Export the trained model
print("Exporting model...") # Visualize user what is happening
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.dirname(__file__)
joblib.dump(model, f"{save_dir}/sklearn_rf_{timestamp}.joblib")

print(f"Model exported as sklearn_rf_{timestamp}.joblib in {save_dir}.")