r"""Script for running thr RWKV model ad-hoc.

Sample usage:

.. code-block:: bash

    python -m rwkv.run 430m "It was the best of times, it was the"
"""

import argparse
from typing import get_args

from ml.utils.logging import configure_logging

from rwkv.model import PretrainedRwkvKey, pretrained_rwkv


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
    args = parser.parse_args()

    configure_logging()

    model = pretrained_rwkv(args.size, empty=args.empty)
    predictor = model.predictor()

    def generate_for_prompt(prompt: str) -> None:
        print(prompt, end="")
        for token in predictor.generate(
            prompt,
            max_len=args.tsz,
            temperature=args.temperature,
            top_p=args.top_p,
            end_strs=args.end_tok,
        ):
            print(token, end=args.sep, flush=True)
        print()

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
