"""Federated learning server package."""

from fl_server.server import FederatedServer
from fl_server.evaluation import evaluate_model
from fl_server.aggregation import aggregate_models_from_files
from fl_server.disagreement import (
    load_disagreements,
    get_active_disagreements,
    create_model_tracks
)

__all__ = [
    "FederatedServer",
    "evaluate_model",
    "aggregate_models_from_files",
    "load_disagreements",
    "get_active_disagreements",
    "create_model_tracks"
]
