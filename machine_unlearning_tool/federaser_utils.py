"""Helpers for the FedEraser-style unlearning path.

Computes per-client parameter deltas across FL rounds, sums/scales them, and
subtracts them from the pre-unlearning model to produce a damaged student
that is then repaired via knowledge distillation on the retain data.
"""

import os
import torch
from typing import List, Dict, Tuple, Optional


def compute_delta_parameters(model_after, model_before):
    """Element-wise ΔM = M_after - M_before. Accepts a model or a state_dict
    on either side."""
    if hasattr(model_after, 'state_dict'):
        params_after = list(model_after.state_dict().values())
    else:
        params_after = list(model_after.values())

    if hasattr(model_before, 'state_dict'):
        params_before = list(model_before.state_dict().values())
    else:
        params_before = list(model_before.values())

    deltas = []
    for p_after, p_before in zip(params_after, params_before):
        delta = p_after.clone() - p_before.clone()
        deltas.append(delta)

    return deltas


def sum_deltas(delta_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Sum multiple delta lists element-wise."""
    if not delta_list:
        return []

    summed = [torch.zeros_like(d) for d in delta_list[0]]
    for deltas in delta_list:
        for i, delta in enumerate(deltas):
            summed[i] += delta

    return summed


def scale_delta(deltas: List[torch.Tensor], scale: float) -> List[torch.Tensor]:
    """Scale every tensor by `scale` (usually 1/N to turn a sum into a mean)."""
    return [d * scale for d in deltas]


def subtract_delta_from_model(model, deltas: List[torch.Tensor]):
    """Subtract each delta from the matching model parameter, in place."""
    params = list(model.parameters())

    if len(params) != len(deltas):
        raise ValueError(f"Parameter count mismatch: model has {len(params)}, deltas has {len(deltas)}")

    for param, delta in zip(params, deltas):
        param.data -= delta

    return model


def get_participated_rounds(results_dir: str, forget_client_id: int) -> List[int]:
    """Rounds where this client actually trained, inferred from the presence of
    its saved model.pt under round_N/clients/client_{id}/."""
    participated = []

    model_storage = os.path.join(results_dir, "model_storage")
    if not os.path.exists(model_storage):
        return participated

    for item in os.listdir(model_storage):
        if item.startswith("round_"):
            try:
                round_num = int(item.split("_")[1])

                client_model_path = os.path.join(
                    model_storage,
                    f"round_{round_num}",
                    "clients",
                    f"client_{forget_client_id}",
                    "model.pt"
                )

                if os.path.exists(client_model_path):
                    participated.append(round_num)

            except (IndexError, ValueError):
                continue

    return sorted(participated)


def compute_client_total_contribution(
    results_dir: str,
    forget_client_id: int,
    device: torch.device,
    experiment_type: str = None
) -> Tuple[List[torch.Tensor], Dict]:
    """Sum a client's parameter contributions over every round it participated in.

    For each round we take (client_model_after - global_model_before) and sum
    across rounds. This feeds the FedEraser update M'F = MF - (1/N) * Σ ΔM.

    Returns (total_delta, metadata); metadata records which rounds were used
    and which were skipped (missing files, load errors).
    """
    metadata = {
        "rounds_processed": [],
        "rounds_skipped": [],
        "total_rounds": 0,
        "delta_norm": 0.0
    }

    participated_rounds = get_participated_rounds(results_dir, forget_client_id)
    metadata["total_rounds"] = len(participated_rounds)

    if not participated_rounds:
        print(f"Warning: Client {forget_client_id} didn't participate in any rounds")
        return [], metadata

    print(f"Computing ΔM for client {forget_client_id} across {len(participated_rounds)} rounds: {participated_rounds}")

    all_deltas = []

    for round_num in participated_rounds:
        try:
            # global model the client started this round with
            global_before_path = os.path.join(
                results_dir,
                "model_storage",
                f"round_{round_num}",
                "global_model_for_training",
                "model.pt"
            )

            # client model after local training
            client_after_path = os.path.join(
                results_dir,
                "model_storage",
                f"round_{round_num}",
                "clients",
                f"client_{forget_client_id}",
                "model.pt"
            )

            if not os.path.exists(global_before_path):
                print(f"  Round {round_num}: Skipping (global_before not found)")
                metadata["rounds_skipped"].append(round_num)
                continue

            if not os.path.exists(client_after_path):
                print(f"  Round {round_num}: Skipping (client_after not found)")
                metadata["rounds_skipped"].append(round_num)
                continue

            global_before = torch.load(global_before_path, map_location=device)
            client_after = torch.load(client_after_path, map_location=device)

            delta_round = compute_delta_parameters(client_after, global_before)
            all_deltas.append(delta_round)

            metadata["rounds_processed"].append(round_num)
            print(f"  Round {round_num}: delta M computed")

        except Exception as e:
            print(f"  Round {round_num}: Error computing delta: {e}")
            metadata["rounds_skipped"].append(round_num)
            continue

    if not all_deltas:
        print("Warning: No deltas computed (all rounds skipped)")
        return [], metadata

    total_delta = sum_deltas(all_deltas)

    delta_norm = sum(torch.norm(d).item() for d in total_delta)
    metadata["delta_norm"] = delta_norm

    print(f"Total ΔM computed: {len(metadata['rounds_processed'])} rounds, norm={delta_norm:.4f}")

    return total_delta, metadata
