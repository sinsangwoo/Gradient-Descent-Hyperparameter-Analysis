"""Command-line interface for PhIO."""

import argparse
import sys
from phio.utils import logger


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PhIO: Physics-Informed Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a PINN model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    train_parser.add_argument("--output", type=str, default="./results", help="Output directory")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained model")
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--output", type=str, default="./eval_results", help="Output directory"
    )

    args = parser.parse_args()

    if args.command == "train":
        logger.info(f"Training with config: {args.config}")
        logger.info("Training command not yet implemented")
        # TODO: Implement training pipeline
    elif args.command == "eval":
        logger.info(f"Evaluating checkpoint: {args.checkpoint}")
        logger.info("Evaluation command not yet implemented")
        # TODO: Implement evaluation pipeline
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
