# RWKV

This repository contains my implementation of the pretrained RWKV model, with numerically-stable Triton kernels and code for doing LoRA fine-tuning.

For an explanation of the math, see [this accompanying blog post](https://ben.bolte.cc/rwkv).

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

## Speed

The following table shows measured tokens per second for generating a sequence on various devices using the `run.py` script. Take these numbers with a heap of salt though because the inference pass is so fast that the overhead of the Python script is significant (I only observed around 15% GPU utilization on my 4090). For really fast inference, I would suggest using [this implementation](https://github.com/saharNooby/rwkv.cpp), which uses GGML.

| Device      | Vanilla | Eps   | Log   |
| ----------- | ------- | ----- | ----- |
| M2 (Metal)  | 30.87   | 26.56 | 26.99 |
| M2 (CPU)    | 61.25   | 58.90 | 60.69 |
| 4090 (BF16) | 101.94  | 99.28 | 93.11 |
| 4090 (Log)  | 100.86  | 96.05 | 92.61 |
