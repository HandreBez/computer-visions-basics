# Computer Vision Basics

Foundational computer vision / CNN work, built from scratch without autocomplete.
This repo is separate from `tb-cxr-classification` on purpose — this is where the
fundamentals get built and tested before they're applied to the actual TB project.

## Principles

- No autocomplete on any from-scratch implementation — type it, feel the resistance,
  push through.
- Understand before implementing — read the theory (CS231n notes, papers) first,
  then build.
- Every exercise should leave behind something reproducible: working code, a short
  writeup of what was learned, and an honest note on what's still unclear.

## Structure

```
computer-vision-basics/
├── cifar10-cnn/       CNN from scratch on CIFAR-10 (architecture, training, filter visualization)
├── notes/             Reading notes on CV papers/articles, full-sentence writeups
└── README.md
```

More exercises will likely get added as folders here over time (e.g. a small ViT
experiment, filter visualization deep-dives) rather than spinning up new repos.

## Log

| Date | What | Status |
|---|---|---|
| 13 July 2026 | CNN shape-logic worked through (channels, padding, pooling, flattening) + model skeleton created | In progress |

## Current focus: `cifar10-cnn/`

Building a CNN from scratch on CIFAR-10 — no pretrained weights, no autocomplete.

- [ ] Define model architecture (`model.py`)
- [ ] Data loading (torchvision CIFAR-10)
- [ ] Training loop
- [ ] Visualize learned filters with matplotlib
- [ ] Short writeup: what worked, what didn't, what's still unclear

See `cifar10-cnn/README.md` for details.