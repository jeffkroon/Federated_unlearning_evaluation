"""Test script for Federated Exact Retraining strategy.

This script tests the TRUE federated unlearning golden standard:
- Replays complete FL process (rounds, local training, FedAvg)
- Excludes forget clients from ALL rounds
- Compares to centralized exact retraining
"""

import os
import sys
import shutil

#Setup path: repo root (fl_orchestrator) + its parent (machine_unlearning_tool)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.dirname(_repo_root))

from fl_orchestrator import FederatedOrchestrator

def main():
    print("="*80)
    print("TESTING FEDERATED EXACT RETRAINING (FL-Native Golden Standard)")
    print("="*80)
    print()
    print("This test will:")
    print("1. Run 3 FL rounds with MNIST (10 clients)")
    print("2. Trigger disagreement in round 2 (Client 0 excludes Client 1)")
    print("3. Apply TWO unlearning strategies:")
    print("   - federated_exact_retraining (replays FL rounds, TRUE FL unlearning)")
    print("   - exact_retraining (centralized pooled training, NOT FL unlearning)")
    print("4. Compare results to show the difference")
    print()
    print("="*80)
    print()

    # Clean up previous results
    if os.path.exists("results/fl_simulation_test"):
        print("Cleaning up previous test results...")
        shutil.rmtree("results/fl_simulation_test")

    # Setup config and disagreements
    config_path = "mock_etcd/test_federated_exact_retraining.json"
    disagreements_path = "mock_etcd/test_federated_disagreements.json"

    # Copy disagreements file to standard location for orchestrator
    shutil.copy(disagreements_path, "mock_etcd/disagreements.json")

    print(f"Using configuration: {config_path}")
    print(f"Using disagreements: {disagreements_path}")
    print()

    #Initialize orchestrator
    print("Initializing Federated Orchestrator...")
    orchestrator = FederatedOrchestrator(config_path=config_path)

    print()
    print("="*80)
    print("STARTING FL TRAINING")
    print("="*80)
    print()

    #Run federated learning
    orchestrator.run_federated_learning()

    print()
    print("="*80)
    print("TEST COMPLETED")
    print("="*80)
    print()
    print("Results saved to:", orchestrator.results_dir)
    print()
    print("To analyze results, check:")
    print("  - Federated Exact Retraining: round_2/tracks/track_0_no1/unlearning/branches/federated_exact_retraining/")
    print("  - Centralized Exact Retraining: round_2/tracks/track_0_no1/unlearning/branches/exact_retraining/")
    print("  - Comparison: round_2/tracks/track_0_no1/unlearning/comparison.json")
    print()
    print("Key difference:")
    print("  - Federated: Replays rounds 1-2 with clients [0,2,3,4,5,6,7,8,9] (excluding 1)")
    print("  - Centralized: Trains on pooled data from same clients (no FL rounds)")
    print()

if __name__ == "__main__":
    main()
