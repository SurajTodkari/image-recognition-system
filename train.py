import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from dataset import get_custom_dataloaders

def train_model():
    # Load the data
    train_loader, test_loader = get_custom_dataloaders()

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    print("Training complete. Model saved as model.pth.")

if __name__ == "__main__":
    train_model()
