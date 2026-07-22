# Import required packages
import os
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score

from src.data_loader import get_sklearn_data

# Load Data
(X_train, y_train), (X_test, y_test) = get_sklearn_data()

# Model A: Multi-layer Perceptron (MLP) Classifier
# a feedforward network with one hidden layer of 128 neurons.
print("Initializing MLP Classifier...")
agent_mlp = MLPClassifier(hidden_layer_sizes=(128,), 
                          max_iter=300,
                          alpha=1e-4,       # L2 penalty (regularization)
                          solver='adam',    # optimization algorithm  
                          random_state=42)

# Model B: Distance-Based Classifier (KNN)
# looks 5 closest images in the training set and predict the label
print("Initializing KNN Classifier...")
agent_knn = KNeighborsClassifier(n_neighbors=5,
                                 weights='distance', 
                                 n_jobs=-1) # n_jobs=-1 uses all available CPU cores for parallel processing

# Model C: Tree-Based Classifier (Random Forest)
# reuse the baseline model
print("Initializing Random Forest Classifier...")
agent_rf = RandomForestClassifier(n_estimators=100, 
                                  random_state=42, 
                                  n_jobs=-1)

# Create multi-agent ensemble
# voting='soft' means it averages the probability from all the agents
print("Building Multi-Agent Voting Ensemble...")
model = VotingClassifier(
    estimators=[
        ('Multi-Layer_Perceptron', agent_mlp),
        ('K_Nearest_Neighbors', agent_knn),
        ('Random_Forest', agent_rf)
    ],
    voting='soft'
)

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
joblib.dump(model, f"{save_dir}/sklearn_ensemble_{timestamp}.joblib")

print(f"Model exported as sklearn_ensemble_{timestamp}.joblib in {save_dir}.")