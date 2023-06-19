# RWKV

This repository contains my implementation of the pretrained RWKV model, with numerically-stable Triton kernels and code for doing LoRA fine-tuning.

For an explanation of the math, see [this accompanying blog post](https://ben.bolte.cc/rwkv).

## WKV Computations

This repo currently contains 3 different implementations of the WKV computation:

1. Vanilla
2. Eps
3. Log

See the blog post linked above for more details about how each of these works.

## Speed

On my M2 Macbook Pro, when running the inference script I get around 30.87 tokens per second for the vanilla WKV computation, while the log computation gets 26.99 tokens per second and the eps computation gets 26.56 tokens per second.
