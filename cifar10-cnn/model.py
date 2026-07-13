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


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: define your conv layers here.
        # Think about in_channels/out_channels chaining between them.
        # e.g. self.conv1 = nn.Conv2d(...)

        # TODO: define your pooling layer(s) here.
        # e.g. self.pool = nn.MaxPool2d(...)

        # TODO: define your final fully-connected (Linear) layer here.
        # Remember: in_features must match your flattened vector size,
        # out_features must be 10.
        # e.g. self.fc = nn.Linear(...)

        pass

    def forward(self, x):
        # TODO: chain the layers together here.
        # Don't forget:
        #   - an activation function (e.g. ReLU) after each conv layer —
        #     we haven't discussed this yet, worth asking about tomorrow
        #     if it's not clear why it's needed
        #   - flattening before the Linear layer (x.view(...) or torch.flatten(...))

        pass


if __name__ == "__main__":
    # Quick sanity check once you've filled in the layers above.
    # A correct model should take a batch of CIFAR-10-shaped images
    # and output a batch of 10-class scores.
    model = CIFAR10CNN()
    dummy_input = torch.randn(1, 3, 32, 32)  # [batch_size, channels, H, W]
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be [1, 10]