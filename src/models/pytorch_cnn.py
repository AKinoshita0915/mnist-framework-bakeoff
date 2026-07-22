# Import required packages
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from src.data_loader import get_pytorch_dataloader

# Parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture
class DigitClassifierCNN(nn.Module):
    def __init__(self):
        super(DigitClassifierCNN, self).__init__()
        # First convolutional layer: input channels = 1 (grayscale), output channels = 16, kernel size = 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional layer: input channels = 16, output channels = 32, kernel size = 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128) # 7x7 is the size after two pooling layers (28/2/2)
        self.fc2 = nn.Linear(128, 10) # 10 output classes for digits 0-9

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # Flatten for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Load Data
train_loader, test_loader = get_pytorch_dataloader(batch_size=BATCH_SIZE)

# Initialize model
model = DigitClassifierCNN().to(DEVICE)

# Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Traning Loop
print("Starting training...") # Visualize user what is happening
for epoch in range(EPOCHS):
    model.train() # Set model to training mode
    running_loss = 0.0
    for img, label in train_loader:
        img, label = img.to(DEVICE), label.to(DEVICE)

        optimizer.zero_grad()   # Clear old gradiants
        outputs = model(img)      # Forward pass
        loss = criterion(outputs, label) # Calculate loss
        loss.backward()         # Backward pass
        optimizer.step()        # Update weights

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval() # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad(): # No need to calculate gradients during evaluation
    for img, label in test_loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

# Export the trained model
print("Exporting model...") # Visualize user what is happening
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.dirname(__file__)
torch.save(model.state_dict(), f"{save_dir}/pytorch_cnn_{timestamp}.pth")

print(f"Model exported as pytorch_cnn_{timestamp}.pth in {save_dir}.")