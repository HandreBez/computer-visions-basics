# CIFAR-10 CNN (from scratch)

A CNN built from scratch on CIFAR-10, no pretrained weights, no autocomplete.
Goal is understanding — of conv layers, padding, pooling, and shape logic —
not chasing state-of-the-art accuracy.

## Shape trace (derived by hand before writing any code)

```
Input:        [3, 32, 32]
Conv + pad:   [16, 32, 32]
Conv + pad:   [32, 32, 32]
MaxPool 2x2:  [32, 16, 16]
Conv + pad:   [64, 16, 16]
MaxPool 2x2:  [64, 8, 8]
Flatten:      [4096]
Linear:       [10]
```

## Files

- `model.py` — model architecture (currently a skeleton, layers not yet filled in)
- `train.py` — training loop (not started)
- `visualize_filters.py` — filter visualization with matplotlib (not started)

## Status

- [x] Shape logic worked through by hand (channels, padding, pooling, flattening)
- [ ] Model architecture filled in
- [ ] Data loading set up (torchvision)
- [ ] Training loop written
- [ ] Model trained, results recorded
- [ ] Filters visualized

## Notes / open questions

- Activation functions (ReLU) not yet covered — needed between conv layers,
  to be added once understood (without them, stacked conv layers collapse to
  one linear operation regardless of depth).
- Data loading not yet touched — need to load CIFAR-10 via torchvision.