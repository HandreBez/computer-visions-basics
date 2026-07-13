# CNN Fundamentals — Shape Logic

*13 July 2026*

These notes cover the core mechanics of how data changes shape as it moves
through a convolutional neural network: channels, padding, pooling, and
flattening. Worked through by hand before writing any code, using CIFAR-10
(32x32x3 images, 10 classes) as the running example.

## Input shape and PyTorch's channel convention

A CIFAR-10 image is 32 pixels tall, 32 pixels wide, with 3 colour channels
(RGB). Conceptually this is easy to picture as height-width-channels, but
PyTorch stores tensors channels-first: `[3, 32, 32]`, not `[32, 32, 3]`. This
is a common source of shape-mismatch bugs for anyone coming from
numpy/matplotlib conventions, so it's worth internalising early rather than
debugging it later.

## What a filter actually is

A convolutional layer contains a set of learnable filters (kernels). Each
filter slides across the input and produces one output value per position,
and the full set of positions for one filter produces one 2D feature map.
The key relationship is: one filter produces one output feature map, and a
layer's `out_channels` is simply the number of filters it has learned.

Filters are not flat 2D grids once a layer sits deeper in the network. A
filter in a layer that receives 16 input channels is itself a 3D block of
shape `[16, 3, 3]` for a 3x3 kernel, because it has to combine information
across every incoming channel to produce its single output feature map. This
means a filter's parameter count is `in_channels x kernel_height x
kernel_width`, and a layer's total parameter count multiplies that by the
number of filters (`out_channels`). Channel depth growing across layers
(e.g. 16 to 32 to 64) is not additive — each layer independently chooses its
own `out_channels`, and deeper layers commonly double the previous depth as
a design convention, not a rule.

The chaining rule that keeps shapes consistent across layers is simple but
easy to get wrong under pressure: a layer's `in_channels` must equal the
previous layer's `out_channels`, because that number is literally the depth
of the data flowing between them.

## Why spatial size shrinks, and how padding controls it

Without padding, a kernel cannot centre itself on the edge pixels of an
image, so the output of a convolution is smaller than the input. For a
kernel sliding with stride 1 and no padding, the number of valid positions
along one dimension is `input_size - kernel_size + 1`, and that number of
valid positions is exactly the output size along that dimension.

Left unmanaged, this shrinkage compounds with every convolutional layer
added, and a deep enough network would eventually run out of spatial size
entirely — quite apart from the fact that it also means real information at
the image's edges gets discarded at every layer, whether or not that's
desired.

Padding — adding a border, usually of zeros, around the input before the
filter slides over it — is the fix. The general formula for output size is
`input_size + 2*padding - kernel_size + 1`. Setting output size equal to
input size and solving gives `padding = (kernel_size - 1) / 2`, which for a
3x3 kernel works out to a padding of 1 pixel on each side (2 pixels added
total to each spatial dimension). This only resolves to a whole number for
odd kernel sizes, which is a real part of why odd kernel sizes (3x3, 5x5,
7x7) are so standard in practice.

Using "same" padding (padding chosen so output size equals input size)
decouples network depth from spatial size — an arbitrary number of
convolutional layers can be stacked without losing pixels or needing to
re-derive the shape at every step.

## Pooling: the deliberate mechanism for shrinking size

If padding keeps spatial size constant indefinitely, something else has to
be responsible for deliberately shrinking it down to something small and
compact before final classification. That mechanism is pooling, most
commonly max pooling.

A max pooling layer with a 2x2 window and stride 2 slides across the input
in non-overlapping 2x2 blocks (stride 2 means it jumps 2 pixels at a time,
rather than 1), and for each block keeps only the single largest value,
discarding the other three. This has two effects: it shrinks spatial size
predictably (a 2x2/stride-2 pool always exactly halves height and width),
and it gives the network a degree of translation invariance, since the
strongest activation in a local region is preserved while its precise
position within that region is not.

The size formula for pooling is the same general shape as for convolution:
`output_size = (input_size - window_size) / stride + 1`. Applied to a 32x32
input with a 2x2 window and stride 2: `(32 - 2) / 2 + 1 = 16`.

## Flattening and the final classification layer

After alternating convolution (with same padding, so size is preserved) and
pooling (which halves size), the network ends up with a small spatial map
carrying many channels — for example `[64, 8, 8]`: 64 channels, each an 8x8
feature map. CIFAR-10 classification needs a final output of 10 numbers, one
raw score per class, so this 3D tensor has to be converted to a flat 1D
vector first. This operation is called flattening, and the resulting vector
length is simply the product of all three dimensions: `64 x 8 x 8 = 4096`.

That flattened vector is fed into a fully connected (linear) layer,
`nn.Linear(4096, 10)`, which maps the 4096 input numbers down to 10 output
scores (logits) — one per class. These logits are what get compared against
the true label during training, typically via cross-entropy loss.

## Full shape trace worked through today

```
Input:        [3, 32, 32]
Conv + pad:   [16, 32, 32]   in_channels=3, out_channels=16, 3x3 kernel, padding=1
Conv + pad:   [32, 32, 32]   in_channels=16, out_channels=32, 3x3 kernel, padding=1
MaxPool 2x2:  [32, 16, 16]
Conv + pad:   [64, 16, 16]   in_channels=32, out_channels=64, 3x3 kernel, padding=1
MaxPool 2x2:  [64, 8, 8]
Flatten:      [4096]
Linear:       [10]
```

## Open questions / not yet covered

- Activation functions (ReLU) were not covered in this session. They sit
  between convolutional layers and are necessary for the network to
  represent anything beyond a single linear transformation, regardless of
  how many layers are stacked — this needs to be understood before the
  model skeleton can actually be filled in and trained.
- Data loading (getting CIFAR-10 into PyTorch via torchvision) has not been
  touched yet.
- Backpropagation through convolutional and pooling layers specifically —
  how gradients actually flow backward through these operations — has not
  been derived yet, only the forward shape logic.