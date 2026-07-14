"""
CIFAR-10 data loading — skeleton only.

Fill in using the reasoning from today's session:
- ToTensor() converts PIL Image (0-255 ints, HWC) -> tensor (0-1 floats, CHW)
- Normalize(mean, std) centers the data using per-channel stats
- Transforms run fresh every __getitem__ call, which is why random
  augmentations (RandomCrop, RandomHorizontalFlip) actually give variety
  across epochs instead of a fixed cached version
- Train and test sets should NOT use the same transforms — augmentation
  (random crop/flip) belongs on the TRAINING set only. Think about why:
  would you want random flips applied to your test/eval images too?

CIFAR-10 per-channel mean/std (commonly used precomputed values, RGB order):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

No autocomplete — type it yourself.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms():
    """
    Return two separate transform pipelines: one for training (with
    augmentation), one for test/eval (no augmentation, just ToTensor +
    Normalize).

    TODO: build train_transform using transforms.Compose([...])
        - RandomCrop(32, padding=4)  -- common CIFAR-10 augmentation choice
        - RandomHorizontalFlip()
        - ToTensor()
        - Normalize(CIFAR10_MEAN, CIFAR10_STD)

    TODO: build test_transform using transforms.Compose([...])
        - ToTensor()
        - Normalize(CIFAR10_MEAN, CIFAR10_STD)
        - (no augmentation here — think about why, from today's discussion)
    """
    train_transform = None  # TODO
    test_transform = None  # TODO

    return train_transform, test_transform


def get_dataloaders(batch_size=64, data_dir="./data"):
    """
    TODO: load CIFAR10 train and test sets via torchvision.datasets.CIFAR10
        - root=data_dir, train=True/False, download=True, transform=...
    TODO: wrap each in a DataLoader
        - train loader should shuffle=True
        - test loader should shuffle=False (no need to shuffle eval data)
    """
    train_transform, test_transform = get_transforms()

    train_set = None  # TODO
    test_set = None  # TODO

    train_loader = None  # TODO
    test_loader = None  # TODO

    return train_loader, test_loader


if __name__ == "__main__":
    # Sanity check once filled in: should print a batch shape like
    # torch.Size([64, 3, 32, 32]) for images and torch.Size([64]) for labels
    train_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)