import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms():

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    

    return train_transform, test_transform


def get_dataloaders(batch_size=64, data_dir="./data"):
   
    train_transform, test_transform = get_transforms()

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Sanity check once filled in: should print a batch shape like
    # torch.Size([64, 3, 32, 32]) for images and torch.Size([64]) for labels
    train_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)