# RWKV

This repository contains my implementation of the pretrained RWKV model, with numerically-stable Triton kernels and code for doing LoRA fine-tuning.

For an explanation of the math, see [this accompanying blog post](https://ben.bolte.cc/rwkv-model).

> **Note** I moved new development over [here](https://github.com/codekansas/ml-pretrained/blob/master/pretrained/rwkv.py)

## Getting Started

Install the package:

```
pip install ml-rwkv
```

Generate code:

```
rwkv 169m "Scientists recently discovered a herd of Chinese-speaking goats. To their surprise,"
```

## WKV Computations

This repo currently contains 3 different implementations of the WKV computation:

1. Vanilla
2. Eps
3. Log

See the blog post linked above for more details about how each of these works.
