#!/usr/bin/env python3
"""
Test script for SISA checkpoint save/load functionality.
This verifies that the metadata-based checkpoint system works correctly.
"""

import os
import sys
import tempfile
import shutil
import torch
import numpy as np
from pathlib import Path

# Add repo root + its parent to path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.dirname(_repo_root))

from fl_server.unlearning_strategies import SISAStrategy
from fl_module.models import create_model


def test_sisa_checkpoint_save_load():
    """Test SISA checkpoint save and load with metadata."""
    print("=" * 80)
    print("Testing SISA Checkpoint System")
    print("=" * 80)

    # Create temporary checkpoint directory
    temp_dir = tempfile.mkdtemp(prefix="sisa_checkpoint_test_")
    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Initialize SISA strategy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        strategy = SISAStrategy(
            model_type="lstm",
            experiment_type="mnist",
            fl_model_params={
                "input_size": 28 * 28,
                "hidden_size": 128,
                "num_layers": 2,
                "num_classes": 10
            },
            num_shards=2,
            num_slices=2,
            checkpoint_dir=temp_dir,
            device=device
        )

        print(f"\nInitialized SISAStrategy with {strategy.num_shards} shards × {strategy.num_slices} slices")

        #Create mock slice models
        mock_models = []
        total_slices = strategy.num_shards * strategy.num_slices

        print(f"\nCreating {total_slices} mock slice models...")
        for shard in range(strategy.num_shards):
            for slice_idx in range(strategy.num_slices):
                #Create a simple model
                model = create_model(
                    experiment_type="mnist",
                    **strategy.fl_model_params
                ).to(device)

                # Create mock IDs for this slice
                slice_ids = set(range(shard * 100 + slice_idx * 25, shard * 100 + (slice_idx + 1) * 25))

                mock_models.append({
                    "model": model,
                    "meta": (shard, slice_idx),
                    "ids": slice_ids
                })
                print(f"  Created slice (shard={shard}, slice={slice_idx}) with {len(slice_ids)} IDs")

        # Test 1: Save models
        print("\n" + "=" * 80)
        print("TEST 1: Saving slice models with metadata")
        print("=" * 80)

        round_num = 5
        strategy._save_slice_models(mock_models, round_num)

        # Verify files exist
        metadata_path = os.path.join(temp_dir, f"sisa_metadata_round{round_num}.json")
        assert os.path.exists(metadata_path), f"Metadata file not found: {metadata_path}"
        print(f"OK: Metadata file created: {metadata_path}")

        #Verify model files exist
        for shard in range(strategy.num_shards):
            for slice_idx in range(strategy.num_slices):
                model_file = f"sisa_slice_s{shard}_sl{slice_idx}_round{round_num}.pt"
                model_path = os.path.join(temp_dir, model_file)
                assert os.path.exists(model_path), f"Model file not found: {model_path}"
                print(f"OK: Model file created: {model_file}")

        #Test 2: Load models
        print("\n" + "=" * 80)
        print("TEST 2: Loading slice models from checkpoint")
        print("=" * 80)

        loaded_models = strategy._load_slice_models(round_num)

        assert loaded_models is not None, "Failed to load models"
        assert len(loaded_models) == total_slices, f"Expected {total_slices} models, got {len(loaded_models)}"
        print(f"OK: Loaded {len(loaded_models)} slice models")

        # Verify structure
        for idx, model_dict in enumerate(loaded_models):
            assert "model" in model_dict, f"Model {idx} missing 'model' key"
            assert "meta" in model_dict, f"Model {idx} missing 'meta' key"
            assert "ids" in model_dict, f"Model {idx} missing 'ids' key"

            shard, slice_idx = model_dict["meta"]
            ids = model_dict["ids"]
            print(f"OK: Slice (shard={shard}, slice={slice_idx}): {len(ids)} IDs")

            # Verify IDs match
            expected_ids = set(range(shard * 100 + slice_idx * 25, shard * 100 + (slice_idx + 1) * 25))
            assert ids == expected_ids, f"IDs mismatch for slice {idx}"

        # Test 3: Model weights preservation
        print("\n" + "=" * 80)
        print("TEST 3: Verifying model weights are preserved")
        print("=" * 80)

        #Set specific weights in original model
        original_model = mock_models[0]["model"]
        with torch.no_grad():
            for param in original_model.parameters():
                param.fill_(42.0)  #Set all weights to 42

        # Save this model
        strategy._save_slice_models(mock_models[:1], round_num=10)

        # Load it back
        loaded = strategy._load_slice_models(round_num=10)
        assert loaded is not None, "Failed to load model"

        loaded_model = loaded[0]["model"]

        # Check weights
        weights_match = True
        for orig_param, loaded_param in zip(original_model.parameters(), loaded_model.parameters()):
            if not torch.allclose(orig_param, loaded_param):
                weights_match = False
                break

        assert weights_match, "Model weights were not preserved correctly"
        print("OK: Model weights preserved correctly")

        #Test 4: Configuration validation
        print("\n" + "=" * 80)
        print("TEST 4: Configuration mismatch detection")
        print("=" * 80)

        #Create strategy with different configuration
        strategy_mismatch = SISAStrategy(
            model_type="lstm",
            experiment_type="mnist",
            fl_model_params=strategy.fl_model_params,
            num_shards=3,  # different shard count
            num_slices=2,
            checkpoint_dir=temp_dir,
            device=device
        )

        # Try to load with mismatched config
        loaded_mismatch = strategy_mismatch._load_slice_models(round_num)
        assert loaded_mismatch is None, "Should return None for config mismatch"
        print("OK: Configuration mismatch detected correctly")

        # Test 5: Missing checkpoint handling
        print("\n" + "=" * 80)
        print("TEST 5: Missing checkpoint handling")
        print("=" * 80)

        loaded_missing = strategy._load_slice_models(round_num=999)
        assert loaded_missing is None, "Should return None for missing checkpoint"
        print("OK: Missing checkpoint handled correctly")

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)

    finally:
        #Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_sisa_checkpoint_save_load()
