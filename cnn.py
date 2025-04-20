import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Device configuration: choose GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 3
log_interval = 200  # How often to print training progress

# Define data transformations: convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize tensors to mean=0.5, std=0.5
])

# Load FashionMNIST dataset (train and test sets)
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# Create DataLoader for batching and shuffling data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN architecture
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # convolution, ReLU activation, and pooling
        x = self.pool(self.relu(self.conv1(x)))  # output shape: [batch, 32, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # output shape: [batch, 64, 7, 7]

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, start_dim=1)  # output shape: [batch, 64*7*7]

        # first fully connected layer and activation
        x = self.relu(self.fc1(x))

        # Apply dropout to prevent overfitting
        x = self.dropout(x)

        # Compute final class logits
        x = self.fc2(x)

        return x

# Instantiate model, loss function, and optimizer
model = FashionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Begin training loop
print("Training started...")
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to device (GPU or CPU)

        # Forward pass: compute predictions
        outputs = model(images)
        # Compute loss between predicted and true labels
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients and update weights
        optimizer.zero_grad()  # Reset gradients from previous iteration
        loss.backward()        # Calculate gradients
        optimizer.step()       # Update parameters based on gradients

        # Periodically evaluate accuracy on test dataset
        if (i + 1) % log_interval == 0:
            model.eval()  # Switch to evaluation mode
            correct = 0
            total = 0
            with torch.no_grad():  # Disable gradient computation for evaluation
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)  # Predicted class is the one with highest score
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            # Print training progress and test accuracy
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%")
            model.train()  # Switch back to training mode

print("Training complete!")
