from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_custom_dataloaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder('./dataset/train', transform=transform)
    test_data = datasets.ImageFolder('./dataset/test', transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
