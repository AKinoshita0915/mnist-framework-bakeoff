# Import required packages
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers, models

from src.data_loader import get_tensorflow_data

# Parameters
BATCH_SIZE = 64
EPOCHS = 5

# Load Data
(x_train, y_train), (x_test, y_test) = get_tensorflow_data()

# Model Architecture
def build_digit_classifier_model():
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)), 

        # Flatten the output for the fully connected layer
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        
        # Output layer: 10 classes (digits 0-9)
        layers.Dense(10)
    ])
    
    return model

# Initialize model
model = build_digit_classifier_model()

# Loss Function & Optimizer
# SparseCategoricalCrossentropy is used for multi-class classification problems where labels are provided as integers rather than one-hot encoded vectors. The 'from_logits=True' argument indicates that the output of the model is not normalized (i.e., it is raw logits).
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
                    loss=loss_fn, 
                    metrics=['accuracy'])

# Traninig model
print("Starting training...") # Visualize user what is happening
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

# Evaluation
print("Evaluating model...") # Visualize user what is happening
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Export the trained model
print("Exporting model...") # Visualize user what is happening
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.dirname(__file__)
model.save(f"{save_dir}/tensorflow_cnn_{timestamp}.keras")

print(f"Model exported as tensorflow_cnn_{timestamp}.keras in {save_dir}.")