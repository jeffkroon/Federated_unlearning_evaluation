"""Command-line interface for federated learning client."""

import argparse
from fl_client.client import FederatedClient

def main():
    """Run the client as a standalone application."""
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--experiment", type=str, default="n_cmapss", choices=["n_cmapss", "mnist", "tabular"], help="Experiment type")
    parser.add_argument("--data_dir", type=str, help="Data directory (defaults to experiment-specific location)")
    parser.add_argument("--sample_size", type=int, default=1000, help="Sample size per client")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--results_dir", type=str, help="Results directory for models and outputs")

    args = parser.parse_args()

    if args.data_dir is None:
        if args.experiment == "n_cmapss":
            args.data_dir = "data/n-cmapss/train"
        elif args.experiment == "mnist":
            args.data_dir = "data/mnist/train"

    client = FederatedClient(
        client_id=args.client_id,
        experiment_type=args.experiment,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        results_dir=args.results_dir
    )

    client.load_data(sample_size=args.sample_size)
    client.train_with_disagreement_resolution()

if __name__ == "__main__":
    main()
