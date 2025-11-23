import argparse
from activation_unlearning.probing.extract import main

def parse_args():
    parser = argparse.ArgumentParser(description="Activation extraction tool")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Provide one or more prompts manually"
    )
    return parser.parse_args()

def run():
    args = parse_args()
    if args.prompt:
        main(prompts=args.prompt)
    else:
        main()

if __name__ == "__main__":
    run()
