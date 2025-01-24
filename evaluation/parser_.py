import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("--data-mode", type=str, required=True, choices=("clean", "unlearn"), default="for model saving")
    parser.add_argument("--file-path", type=str, required=True, help="path to dir containing train.csv")
    parser.add_argument("--test-file-path", type=str, required=True, help="path to dir containing test.csv (clean)")
    parser.add_argument("--hf-token", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=10) # sufficient generally for polarity, change as needed 
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument(
        "--model-name", 
        type=str, required=True, 
        choices=(
            "mistral-7b-v0.3-bnb-4bit",
            "mistral-7b-instruct-v0.3-bnb-4bit",
            "Meta-Llama-3.1-8B-bnb-4bit",
            "Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "Meta-Llama-3.1-70B-bnb-4bit",
            "Meta-Llama-3.1-70B-Instruct-bnb-4bit"
        )
    )

    return parser.parse_args()