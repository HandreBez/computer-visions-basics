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

- `data.py` — CIFAR-10 loading via torchvision, train/test transforms
- `model.py` — model architecture (filled in, matches shape trace above)
- `train.py` — training loop (not started)
- `visualize_filters.py` — filter visualization with matplotlib (not started)

## Status

- [x] Shape logic worked through by hand (channels, padding, pooling, flattening)
- [x] Model architecture filled in
- [x] Data loading set up (torchvision)
- [ ] Training loop written
- [ ] Model trained, results recorded
- [ ] Filters visualized

## Notes / open questions

- Activation functions (ReLU) — now used between conv layers in the
  architecture; without them, stacked conv layers collapse to one linear
  operation regardless of depth.