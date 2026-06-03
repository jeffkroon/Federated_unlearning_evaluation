"""
Unit tests for FedEraser utilities.

Tests delta computation, summation, scaling, and model manipulation functions.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


def test_compute_delta_parameters():
    """Test delta computation between two models."""
    from machine_unlearning_tool.federaser_utils import compute_delta_parameters

    # Create two simple state dicts
    state_dict_before = {
        'layer1.weight': torch.ones(5, 5),
        'layer1.bias': torch.zeros(5),
        'layer2.weight': torch.ones(3, 5),
        'layer2.bias': torch.zeros(3)
    }

    state_dict_after = {
        'layer1.weight': torch.ones(5, 5) * 2,  # Changed
        'layer1.bias': torch.ones(5),  # Changed
        'layer2.weight': torch.ones(3, 5) * 2,  #Changed
        'layer2.bias': torch.ones(3)  #Changed
    }

    # Compute delta
    deltas = compute_delta_parameters(state_dict_after, state_dict_before)

    # Verify delta is correct (should all be ones)
    assert len(deltas) == 4
    assert torch.allclose(deltas[0], torch.ones(5, 5), atol=1e-6)
    assert torch.allclose(deltas[1], torch.ones(5), atol=1e-6)
    assert torch.allclose(deltas[2], torch.ones(3, 5), atol=1e-6)
    assert torch.allclose(deltas[3], torch.ones(3), atol=1e-6)


def test_sum_deltas():
    """Test delta summation."""
    from machine_unlearning_tool.federaser_utils import sum_deltas

    delta1 = [torch.ones(5, 5), torch.ones(5)]
    delta2 = [torch.ones(5, 5) * 2, torch.ones(5) * 2]
    delta3 = [torch.ones(5, 5) * 3, torch.ones(5) * 3]

    summed = sum_deltas([delta1, delta2, delta3])

    assert len(summed) == 2
    assert torch.allclose(summed[0], torch.ones(5, 5) * 6, atol=1e-6)
    assert torch.allclose(summed[1], torch.ones(5) * 6, atol=1e-6)


def test_sum_deltas_empty():
    """Test delta summation with empty list."""
    from machine_unlearning_tool.federaser_utils import sum_deltas

    summed = sum_deltas([])
    assert summed == []


def test_scale_delta():
    """Test delta scaling."""
    from machine_unlearning_tool.federaser_utils import scale_delta

    deltas = [torch.ones(5, 5), torch.ones(5)]

    # Scale by 0.5
    scaled = scale_delta(deltas, 0.5)

    assert len(scaled) == 2
    assert torch.allclose(scaled[0], torch.ones(5, 5) * 0.5, atol=1e-6)
    assert torch.allclose(scaled[1], torch.ones(5) * 0.5, atol=1e-6)

    #Scale by 1/10 (simulating N=10 clients)
    scaled = scale_delta(deltas, 1.0 / 10)

    assert torch.allclose(scaled[0], torch.ones(5, 5) * 0.1, atol=1e-6)
    assert torch.allclose(scaled[1], torch.ones(5) * 0.1, atol=1e-6)


def test_subtract_delta_from_model():
    """Test subtracting delta from model parameters."""
    from machine_unlearning_tool.federaser_utils import subtract_delta_from_model
    import torch.nn as nn

    #Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(5, 3)
            # Initialize with ones
            with torch.no_grad():
                self.layer1.weight.fill_(1.0)
                self.layer1.bias.fill_(1.0)

        def forward(self, x):
            return self.layer1(x)

    model = SimpleModel()

    # Create delta of 0.5 for all parameters
    deltas = [torch.ones_like(p) * 0.5 for p in model.parameters()]

    # Subtract delta
    model_modified = subtract_delta_from_model(model, deltas)

    #Verify subtraction (should be 1.0 - 0.5 = 0.5)
    for param in model_modified.parameters():
        assert torch.allclose(param, torch.ones_like(param) * 0.5, atol=1e-6)

    #Verify model is modified in-place
    assert model_modified is model


def test_subtract_delta_parameter_mismatch():
    """Test that parameter count mismatch raises error."""
    from machine_unlearning_tool.federaser_utils import subtract_delta_from_model
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(5, 3)

        def forward(self, x):
            return self.layer1(x)

    model = SimpleModel()

    # Create delta with wrong number of parameters
    deltas = [torch.ones(5, 3)]  # Only one delta, but model has 2 parameters (weight + bias)

    with pytest.raises(ValueError, match="Parameter count mismatch"):
        subtract_delta_from_model(model, deltas)


def test_get_participated_rounds_no_directory():
    """Test get_participated_rounds with non-existent directory."""
    from machine_unlearning_tool.federaser_utils import get_participated_rounds

    # Non-existent directory
    rounds = get_participated_rounds("/non/existent/path", forget_client_id=0)
    assert rounds == []


def test_compute_client_total_contribution_no_rounds():
    """Test compute_client_total_contribution with no participated rounds."""
    from machine_unlearning_tool.federaser_utils import compute_client_total_contribution

    device = torch.device("cpu")

    #Non-existent directory
    delta, metadata = compute_client_total_contribution(
        results_dir="/non/existent/path",
        forget_client_id=0,
        device=device
    )

    assert delta == []
    assert metadata["total_rounds"] == 0
    assert metadata["rounds_processed"] == []
    assert metadata["rounds_skipped"] == []
    assert metadata["delta_norm"] == 0.0


def test_federaser_full_pipeline():
    """Test the full FedEraser delta computation pipeline."""
    from machine_unlearning_tool.federaser_utils import (
        compute_delta_parameters,
        sum_deltas,
        scale_delta,
        subtract_delta_from_model
    )
    import torch.nn as nn

    #Simulate FL scenario:
    # - 10 clients
    # - Client 0 participates in 3 rounds
    # - Each round: client model improves by adding 0.1 to all parameters

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(5, 3)

        def forward(self, x):
            return self.layer1(x)

    #Global model (MF) - initialize with ones
    mf_model = SimpleModel()
    with torch.no_grad():
        for p in mf_model.parameters():
            p.fill_(1.0)

    #Simulate 3 rounds of client updates
    num_rounds = 3
    all_deltas = []

    for round_idx in range(num_rounds):
        # Global model before training (what client started with)
        global_before = SimpleModel()
        with torch.no_grad():
            for p in global_before.parameters():
                p.fill_(1.0)  # Simplified: global model stays constant

        # Client model after training (improved)
        client_after = SimpleModel()
        with torch.no_grad():
            for p in client_after.parameters():
                p.fill_(1.0 + 0.1 * (round_idx + 1))  #Incremental improvement

        #Compute delta for this round
        delta_round = compute_delta_parameters(client_after, global_before)
        all_deltas.append(delta_round)

    # Sum all deltas
    total_delta = sum_deltas(all_deltas)

    # Expected total delta: round 0 (0.1) + round 1 (0.2) + round 2 (0.3) = 0.6
    for d in total_delta:
        assert torch.allclose(d, torch.ones_like(d) * 0.6, atol=1e-6)

    # Scale by 1/N (N=10 clients)
    scaled_delta = scale_delta(total_delta, 1.0 / 10)

    #Expected: 0.6 / 10 = 0.06
    for d in scaled_delta:
        assert torch.allclose(d, torch.ones_like(d) * 0.06, atol=1e-6)

    #Compute M'F = MF - (1/N) * Σ(ΔM)
    mf_damaged = subtract_delta_from_model(mf_model, scaled_delta)

    # Expected: 1.0 - 0.06 = 0.94
    for p in mf_damaged.parameters():
        assert torch.allclose(p, torch.ones_like(p) * 0.94, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
