from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(train_size=35000, val_size=10000, test_size=5000, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset, test_dataset = random_split(cifar10_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
