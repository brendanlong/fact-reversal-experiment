"""Main entry point for the fact reversal experiment."""

import argparse
import torch
from fact_reversal.data import generate_facts, split_data
from fact_reversal.models import SingleLayerFull, SingleLayerHalf, TwoLayerNet
from fact_reversal.train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Run the fact reversal experiment"
    )
    parser.add_argument(
        "--num_facts", type=int, default=10, help="Number of facts to generate"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    print(f"Generating {args.num_facts} facts...")
    facts = generate_facts(args.num_facts, seed=args.seed)
    print(f"Generated {len(facts)} facts")

    print("Splitting data into train/val/test...")
    train_data, val_data, test_data = split_data(facts, seed=args.seed)
    print(
        f"Train: {len(train_data.samples)}, "
        f"Val: {len(val_data.samples)}, "
        f"Test: {len(test_data.samples)}"
    )

    vocab_size = train_data.vocab_size

    models = {
        "SingleLayerFull": SingleLayerFull(vocab_size),
        "SingleLayerHalf": SingleLayerHalf(vocab_size),
        "TwoLayerNet": TwoLayerNet(vocab_size),
    }

    print(f"\nVocabulary size: {vocab_size}")
    print("\nTraining models...")

    results = {}

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params}")

        history = train_model(
            model,
            train_data,
            val_data,
            test_data,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )

        results[model_name] = {
            "model": model,
            "history": history,
        }

        # Print final results
        final_train = history["train"][-1]
        final_test = history["test"][-1]

        print(f"\nFinal Results:")
        print(f"  Train Accuracy: {final_train.accuracy:.4f}")
        print(f"    - Forward: {final_train.forward_accuracy:.4f}")
        print(f"    - Reverse: {final_train.reverse_accuracy:.4f}")
        print(f"  Test Accuracy: {final_test.accuracy:.4f}")
        print(f"    - Forward: {final_test.forward_accuracy:.4f}")
        print(f"    - Reverse: {final_test.reverse_accuracy:.4f}")
        print(f"  Test Loss: {final_test.loss:.4f}")

    print(f"\n{'='*60}")
    print("Experiment Complete")
    print(f"{'='*60}")

    # Summary comparison
    print("\nModel Comparison (Test Set):")
    print(f"{'Model':<20} {'Accuracy':<12} {'Forward':<12} {'Reverse':<12}")
    print("-" * 56)

    for model_name, result in results.items():
        final_test = result["history"]["test"][-1]
        print(
            f"{model_name:<20} {final_test.accuracy:<12.4f} "
            f"{final_test.forward_accuracy:<12.4f} "
            f"{final_test.reverse_accuracy:<12.4f}"
        )


if __name__ == "__main__":
    main()
