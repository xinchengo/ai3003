# argparse for handling command-line arguments
import argparse
from trainer import train_pipeline, evaluate

import modal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        "Helper script for AI3003 lab3: Sentiment Analysis\n")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"],
                        help="Mode to run: train or eval")
    parser.add_argument("--config", type=str, required=True,
                        help="Name of the config in config.json to use")
    parser.add_argument("--path", type=str, help="Optional argument to specify"
                        " checkpoint path for evaluation")

    args = parser.parse_args()

    if args.mode == "train":
        checkpoint_path = train_pipeline(args.config)
        print(f"\nTraining completed. Checkpoint saved at: {checkpoint_path}")
    elif args.mode == "eval":
        if not args.path:
            raise ValueError("--path is required for evaluation mode")
        evaluate(args.config, args.path)