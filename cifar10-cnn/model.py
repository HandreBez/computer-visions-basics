"""
CIFAR-10 CNN — skeleton only.

Fill in the layers yourself using the reasoning from today's session:
- in_channels / out_channels chaining between conv layers
- padding=1 for 3x3 kernels to keep spatial size constant ("same" padding)
- MaxPool2d(2, 2) halves spatial size each time
- flatten before the final Linear layer
- Linear layer's out_features must be 10 (CIFAR-10 has 10 classes)

Reference shape trace we derived together (feel free to change the exact
channel counts, but the SHAPE LOGIC should follow this pattern):

    Input:        [3, 32, 32]
    Conv + pad:   [16, 32, 32]
    Conv + pad:   [32, 32, 32]
    MaxPool 2x2:  [32, 16, 16]
    Conv + pad:   [64, 16, 16]
    MaxPool 2x2:  [64, 8, 8]
    Flatten:      [4096]
    Linear:       [10]

No autocomplete — type it yourself, even the parts that feel obvious.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: define your conv layers here.
        # Think about in_channels/out_channels chaining between them.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # TODO: define your pooling layer(s) here.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # TODO: define your final fully-connected (Linear) layer here.
        # Remember: in_features must match your flattened vector size,
        # out_features must be 10.
        self.fc = nn.Linear(in_features=4096, out_features=10)

        pass

    def forward(self, x):
        
        #Layer 1
        x = self.conv1(x)
        x = F.relu(x)

        #Layer2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        #Layer 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        #Flattening
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    # Quick sanity check once you've filled in the layers above.
    # A correct model should take a batch of CIFAR-10-shaped images
    # and output a batch of 10-class scores.
    model = CIFAR10CNN()
    dummy_input = torch.randn(1, 3, 32, 32)  # [batch_size, channels, H, W]
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be [1, 10]