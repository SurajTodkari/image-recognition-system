import torch
from model import SimpleCNN  # Updated import
from dataset import get_custom_dataloaders

def test_model():
    _, test_loader = get_custom_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)  # Updated class name
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    test_model()
