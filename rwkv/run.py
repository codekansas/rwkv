r"""Script for running thr RWKV model ad-hoc.

Sample usage:

.. code-block:: bash

    rwkv 430m "It was the best of times, it was the"
"""

import argparse
import time
from typing import get_args

from ml.utils.logging import configure_logging

from rwkv.model import PretrainedRwkvKey, pretrained_rwkv
from rwkv.wkv import WkvImpl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=str, choices=get_args(PretrainedRwkvKey))
    parser.add_argument("prompt", type=str, nargs="?")
    parser.add_argument("-t", "--tsz", type=int, default=128)
    parser.add_argument("-m", "--temperature", type=float, default=1.0)
    parser.add_argument("-p", "--top-p", type=float, default=0.85)
    parser.add_argument("-e", "--end-tok", type=str, nargs="+", default=[])
    parser.add_argument("-s", "--sep", type=str, default="")
    parser.add_argument("-y", "--empty", action="store_true")
    parser.add_argument("-w", "--wkv-impl", type=str, choices=get_args(WkvImpl))
    args = parser.parse_args()

    configure_logging()

    model = pretrained_rwkv(args.size, empty=args.empty, wkv_impl=args.wkv_impl)
    predictor = model.predictor()

    def generate_for_prompt(prompt: str) -> None:
        print(prompt, end="")
        start_time: float | None = None
        num_tokens = 0
        for token in predictor.generate(
            prompt,
            max_len=args.tsz,
            temperature=args.temperature,
            top_p=args.top_p,
            end_strs=args.end_tok,
        ):
            print(token, end=args.sep, flush=True)
            if start_time is None:
                start_time = time.time()
            num_tokens += 1
        print()
        end_time = time.time()
        if start_time is not None:
            time_delta = end_time - start_time
            print(f"Time taken: {num_tokens} / {time_delta:.2f}s = {num_tokens / time_delta:.2f} tokens per second")

    if args.prompt:
        generate_for_prompt(args.prompt)

    else:
        prompt = input("Prompt: ")
        while prompt:
            generate_for_prompt(prompt)
            prompt = input("Prompt: ")


if __name__ == "__main__":
    # python -m rwkv.run
    main()
