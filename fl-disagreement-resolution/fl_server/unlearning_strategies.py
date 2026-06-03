"""Unlearning strategy implementations for federated learning."""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
machine_unlearning_path = os.path.join(base_dir, "machine_unlearning_tool")
if os.path.exists(machine_unlearning_path):
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

_machine_unlearning_available = False
try:
    from machine_unlearning_tool import (
        run_exact_retraining,
        run_sisa_unlearning,
        run_knowledge_distillation,
        create_model,
        is_pytorch_model,
    )
    _machine_unlearning_available = True
except ImportError as e:
    # If the tool isn't importable, log the paths so it's quick to debug.
    # The strategies themselves check _machine_unlearning_available later
    #and fail with a clean error.
    print(f"Warning: could not import machine_unlearning_tool: {e}")
    print(f"  base_dir={base_dir}, tool_path={machine_unlearning_path}, "
          f"exists={os.path.exists(machine_unlearning_path)}, "
          f"on_sys_path={base_dir in sys.path}")

    def is_pytorch_model(model):
        return isinstance(model, torch.nn.Module)


class UnlearningStrategy(ABC):
    """Abstract base class for unlearning strategies."""
    
    def __init__(
        self,
        model_type: str = "lstm",
        model_params: Optional[Dict] = None,
        train_params: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        experiment_type: Optional[str] = None,
        fl_model_params: Optional[Dict] = None,
        results_dir: Optional[str] = None,
        num_clients: Optional[int] = None
    ):
        """Configure a strategy. `experiment_type` + `fl_model_params` are used
        (where needed) to rebuild the exact same model architecture as the
        FL server, so unlearning runs on the matching architecture.
        `results_dir` and `num_clients` are only relevant for FedEraser:
        delta computation and the 1/N scaling step."""
        self.model_type = model_type
        self.model_params = model_params or {}
        self.train_params = train_params or {}
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.experiment_type = experiment_type
        self.fl_model_params = fl_model_params or {}
        self.results_dir = results_dir
        self.num_clients = num_clients
    
    @abstractmethod
    def unlearn(
        self,
        pretrained_model: Any,
        all_data: Dict[str, Any],
        forget_ids: List[int],
        round_num: int,
        test_loader: Optional[Any] = None,
        test_df: Optional[pd.DataFrame] = None,
        test_input_cols: Optional[List[str]] = None,
        test_target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the unlearning strategy and return the resulting model + metrics.
        `test_loader` is for PyTorch models; `test_df` + `test_input_cols` /
        `test_target_col` for sklearn models. Subclasses typically only use
        one of these two paths."""
        pass


class ExactRetrainingStrategy(UnlearningStrategy):
    """Exact retraining strategy (ground truth)."""
    
    def unlearn(
        self,
        pretrained_model: Any,
        all_data: Dict[str, Any],
        forget_ids: List[int],
        round_num: int,
        test_loader: Optional[Any] = None,
        test_df: Optional[pd.DataFrame] = None,
        test_input_cols: Optional[List[str]] = None,
        test_target_col: Optional[str] = None,
        baseline_model: Optional[Any] = None,
        baseline_original_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Apply exact retraining unlearning."""
        print(f"\n[EXACT RETRAINING] Starting unlearning for {len(forget_ids)} client(s)")
        print(f"  Forget IDs: {sorted(list(forget_ids))}")
        print(f"  Experiment type: {self.experiment_type}")

        start_time = time.time()

        #Extract data
        df = all_data["df"]
        X = all_data["X"]
        y = all_data["y"]
        input_cols = all_data["input_cols"]
        target_col = all_data["target_col"]
        id_column = all_data["id_column"]
        seq_len = all_data.get("seq_len", 24)
        
        if not _machine_unlearning_available:
            raise ImportError(
                "machine_unlearning_tool is not available. "
                "Please ensure all dependencies (including xgboost) are installed and "
                "the machine_unlearning_tool module is accessible."
            )
        
        # Train new model on retained data (same architecture as FL)
        
        # Import FL model factory
        from fl_module import create_model as create_fl_model

        fl_model_params = self.fl_model_params.copy() if self.fl_model_params else {}

        # pull model dims off the pretrained model when not already set
        if pretrained_model is not None:
            if hasattr(pretrained_model, 'input_dim') and 'input_dim' not in fl_model_params:
                fl_model_params['input_dim'] = pretrained_model.input_dim
            if hasattr(pretrained_model, 'output_dim') and 'output_dim' not in fl_model_params:
                fl_model_params['output_dim'] = pretrained_model.output_dim
            if hasattr(pretrained_model, 'hidden_dim') and 'hidden_dim' not in fl_model_params:
                fl_model_params['hidden_dim'] = pretrained_model.hidden_dim
            if hasattr(pretrained_model, 'hidden_dims') and 'hidden_dims' not in fl_model_params:
                fl_model_params['hidden_dims'] = pretrained_model.hidden_dims
            if hasattr(pretrained_model, 'dropout') and 'dropout' not in fl_model_params:
                fl_model_params['dropout'] = pretrained_model.dropout
        
        #Always start from a fresh model (full retrain on retain-set only)
        pretrained = None
        if self.experiment_type:
            new_model = create_fl_model(
                experiment_type=self.experiment_type,
                **fl_model_params
            ).to(self.device)
            pretrained = new_model
        else:
            #Determine input_size from data shape
            model_params = self.model_params.copy() if self.model_params else {}
            if self.model_type == "mlp" and "input_size" not in model_params:
                if X.shape[1] > 0:
                    model_params["input_size"] = X.shape[1]
                else:
                    model_params["input_size"] = len(input_cols)
        
        # Run exact retraining
        result = run_exact_retraining(
            X=X,
            y=y,
            df=df,
            input_cols=input_cols,
            target_col=target_col,
            id_column=id_column,
            forget_ids=forget_ids,
            device=str(self.device),
            seq_len=seq_len,
            model_params=self.model_params if not self.experiment_type else {},
            train_params=self.train_params,
            pretrained_model=pretrained,
            model_type=self.model_type,
            experiment_type=self.experiment_type, 
            fl_model_params=fl_model_params,
            test_df=test_df,
            test_input_cols=test_input_cols,
            test_target_col=test_target_col,
            test_loader=test_loader,
            baseline_original_model=baseline_original_model
        )
        
        unlearning_time = time.time() - start_time
        
        # Extract metrics
        metrics = {}
        metrics.update(result.get("metrics_utility", {}))
        for key, val in result.get("metrics_retain", {}).items():
            metrics.setdefault(key, val)
        metrics["unlearning_time_s"] = unlearning_time
        
        # Add forget-set metrics if available
        if "metrics_forget" in result:
            metrics.update(result["metrics_forget"])
        
        #Add efficiency metrics if available
        if "efficiency_metrics" in result:
            metrics.update(result["efficiency_metrics"])
        
        #Add behavioral distance metrics if available
        if "behavioral_distance" in result:
            metrics.update(result["behavioral_distance"])
        metrics["unlearned"] = True
        
        return {
            "model": result["model"],
            "metrics": metrics
        }


class SISAStrategy(UnlearningStrategy):
    """SISA (Sharding, Isolation, Slicing, Aggregation) strategy."""
    
    def __init__(
        self,
        model_type: str = "lstm",
        model_params: Optional[Dict] = None,
        train_params: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        experiment_type: Optional[str] = None,
        fl_model_params: Optional[Dict] = None,
        num_shards: int = 2,
        num_slices: int = 2,
        checkpoint_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        num_clients: Optional[int] = None
    ):
        """checkpoint_dir lets SISA store/reuse slice models across rounds."""
        super().__init__(
            model_type=model_type,
            model_params=model_params,
            train_params=train_params,
            device=device,
            experiment_type=experiment_type,
            fl_model_params=fl_model_params,
            results_dir=results_dir,
            num_clients=num_clients
        )
        self.num_shards = num_shards
        self.num_slices = num_slices
        self.checkpoint_dir = checkpoint_dir
        self._slice_models_cache = None  # Cache for loaded slice models

    def _save_slice_models(self, models: List[Dict], round_num: int) -> None:
        """Write each SISA slice model + a metadata file to the checkpoint dir."""
        if self.checkpoint_dir is None:
            return  # nothing to do without a checkpoint dir

        import os
        import json
        from datetime import datetime

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        metadata = {
            "version": "1.0",
            "checkpoint_type": "sisa",
            "round_num": round_num,
            "created_at": datetime.now().isoformat(),
            "config": {
                "num_shards": self.num_shards,
                "num_slices": self.num_slices,
                "total_slices": len(models),
                "model_type": self.model_type,
                "experiment_type": self.experiment_type
            },
            "slices": []
        }

        # Save each slice model and record in metadata
        for idx, model_dict in enumerate(models):
            model = model_dict['model']
            shard, slice_idx = model_dict['meta']
            ids = list(model_dict['ids'])  #Convert set to list for JSON serialization

            if is_pytorch_model(model):
                #Save PyTorch model state dict
                filename = f"sisa_slice_s{shard}_sl{slice_idx}_round{round_num}.pt"
                filepath = os.path.join(self.checkpoint_dir, filename)
                torch.save(model.state_dict(), filepath)
            else:
                # Save sklearn model
                import joblib
                filename = f"sisa_slice_s{shard}_sl{slice_idx}_round{round_num}.pkl"
                filepath = os.path.join(self.checkpoint_dir, filename)
                joblib.dump(model, filepath)

            # Add slice metadata
            metadata["slices"].append({
                "index": idx,
                "shard": shard,
                "slice": slice_idx,
                "ids": ids,
                "n_samples": len(ids),
                "model_file": filename
            })

        # Save metadata file
        metadata_path = os.path.join(self.checkpoint_dir, f"sisa_metadata_round{round_num}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {len(models)} SISA slice models + metadata to {self.checkpoint_dir}")

    def _load_slice_models(self, round_num: int) -> Optional[List[Dict]]:
        """Load saved SISA slice models for a round; None if missing or config mismatch."""
        if self.checkpoint_dir is None:
            return None

        import os
        import json

        if not os.path.exists(self.checkpoint_dir):
            return None

        metadata_path = os.path.join(self.checkpoint_dir, f"sisa_metadata_round{round_num}.json")
        if not os.path.exists(metadata_path):
            print(f"No SISA metadata found at {metadata_path}")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        config = metadata["config"]
        if config["num_shards"] != self.num_shards or config["num_slices"] != self.num_slices:
            print(f"Warning: Checkpoint config mismatch. Expected {self.num_shards}x{self.num_slices}, "
                  f"got {config['num_shards']}x{config['num_slices']}")
            return None

        if config.get("model_type") != self.model_type:
            print(f"Warning: Model type mismatch. Expected {self.model_type}, got {config.get('model_type')}")
            return None

        loaded_models = []
        for slice_meta in metadata["slices"]:
            model_file = slice_meta["model_file"]
            filepath = os.path.join(self.checkpoint_dir, model_file)

            if not os.path.exists(filepath):
                print(f"Warning: Slice model file missing: {filepath}")
                return None

            if filepath.endswith('.pt'):
                from fl_module.models import create_model as create_fl_model

                if self.experiment_type:
                    model = create_fl_model(
                        experiment_type=self.experiment_type,
                        **self.fl_model_params
                    ).to(self.device)
                else:
                    from machine_unlearning_tool.model_utils import create_model
                    model = create_model(self.model_type, **self.model_params)
                    if hasattr(model, 'to'):
                        model = model.to(self.device)

                state_dict = torch.load(filepath, map_location=self.device)
                model.load_state_dict(state_dict)
            else:
                import joblib
                model = joblib.load(filepath)

            loaded_models.append({
                "model": model,
                "meta": (slice_meta["shard"], slice_meta["slice"]),
                "ids": set(slice_meta["ids"])
            })

        print(f"Loaded {len(loaded_models)} SISA slice models from checkpoint (round {round_num})")
        return loaded_models
    
    def unlearn(
        self,
        pretrained_model: Any,
        all_data: Dict[str, Any],
        forget_ids: List[int],
        round_num: int,
        test_loader: Optional[Any] = None,
        test_df: Optional[pd.DataFrame] = None,
        test_input_cols: Optional[List[str]] = None,
        test_target_col: Optional[str] = None,
        baseline_model: Optional[Any] = None,
        baseline_original_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Apply SISA unlearning."""
        print(f"\n[SISA] Starting unlearning for {len(forget_ids)} client(s)")
        print(f"  Forget IDs: {sorted(list(forget_ids))}")
        print(f"  Shards: {self.num_shards}, Slices: {self.num_slices}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")

        start_time = time.time()

        #Extract data
        df = all_data["df"]
        X = all_data["X"]
        y = all_data["y"]
        input_cols = all_data["input_cols"]
        target_col = all_data["target_col"]
        id_column = all_data["id_column"]
        seq_len = all_data.get("seq_len", 24)
        
        if not _machine_unlearning_available:
            raise ImportError(
                "machine_unlearning_tool is not available. "
                "Please ensure all dependencies (including xgboost) are installed and "
                "the machine_unlearning_tool module is accessible."
            )
        
        #Convert pretrained model
        pretrained = pretrained_model
        
        # load pretrained slice models (reuse unaffected slices)
        total_slices = self.num_shards * self.num_slices
        pretrained_models = self._load_slice_models(round_num)

        if pretrained_models is not None:
            print(f"Loaded {len(pretrained_models)} SISA slice models from checkpoint (round {round_num})")
        else:
            print(f"No SISA checkpoints found - training all {total_slices} slices from scratch")
        
        # eval against baseline_model (Exact RT) if set, else pretrained_model
        baseline_for_eval = baseline_model if baseline_model is not None else pretrained
        
        # Run SISA unlearning
        result = run_sisa_unlearning(
            X=X,
            y=y,
            df=df,
            input_cols=input_cols,
            target_col=target_col,
            id_column=id_column,
            forget_ids=forget_ids,
            device=str(self.device),
            seq_len=seq_len,
            num_shards=self.num_shards,
            num_slices=self.num_slices,
            model_params=self.model_params,
            train_params=self.train_params,
            pretrained_models=pretrained_models,
            pretrained_model=pretrained,  #Still use pretrained for training
            baseline_model=baseline_for_eval,  #Use baseline (Exact Retraining) for evaluation
            baseline_original_model=baseline_original_model,
            model_type=self.model_type,
            experiment_type=self.experiment_type,
            fl_model_params=self.fl_model_params,
            test_loader=test_loader,
            test_df=test_df,
            test_input_cols=test_input_cols,
            test_target_col=test_target_col
        )
        
        unlearning_time = time.time() - start_time
        
        # Save slice models for future reuse
        if result.get("models") and self.checkpoint_dir is not None:
            self._save_slice_models(result["models"], round_num)

        # Get ensemble model
        ensemble_model = result.get("ensemble", result["models"][0]["model"] if result.get("models") else None)
        
        # Start with retain metrics (utility preservation)
        #Metrics are calculated on the ensemble model, not approximated
        metrics = {}
        metrics.update(result.get("metrics_utility", {}))
        for key, val in result.get("metrics_retain", {}).items():
            metrics.setdefault(key, val)
        metrics["unlearning_time_s"] = unlearning_time
        
        #Add forget-set metrics if available
        if "metrics_forget" in result:
            metrics.update(result["metrics_forget"])
        
        # Add efficiency metrics if available
        if "efficiency_metrics" in result:
            metrics.update(result["efficiency_metrics"])
        else:
            # Fallback: try to extract from models if available
            if result.get("models") and len(result["models"]) > 0:
                first_model_metrics = result["models"][0].get("metrics", {})
                if "retrain_fraction" in first_model_metrics:
                    metrics["retrain_fraction"] = first_model_metrics["retrain_fraction"]
                    metrics["N_train_total"] = first_model_metrics.get("N_train_total", 0)
                    metrics["N_retrain"] = first_model_metrics.get("N_retrain", 0)
                    metrics["num_affected_slices"] = first_model_metrics.get("num_affected_slices", 0)
                    metrics["total_slices"] = first_model_metrics.get("total_slices", 0)
        
        # Add behavioral distance metrics if available
        if "behavioral_distance" in result:
            metrics.update(result["behavioral_distance"])
        
        return {
            "model": ensemble_model,
            "metrics": metrics
        }


class DistillationStrategy(UnlearningStrategy):
    """Knowledge distillation strategy."""
    
    def unlearn(
        self,
        pretrained_model: Any,
        all_data: Dict[str, Any],
        forget_ids: List[int],
        round_num: int,
        test_loader: Optional[Any] = None,
        test_df: Optional[pd.DataFrame] = None,
        test_input_cols: Optional[List[str]] = None,
        test_target_col: Optional[str] = None,
        baseline_model: Optional[Any] = None,
        baseline_original_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Apply knowledge distillation unlearning."""
        print(f"\n[KNOWLEDGE DISTILLATION] Starting unlearning for {len(forget_ids)} client(s)")
        print(f"  Forget IDs: {sorted(list(forget_ids))}")
        #Temperature and alpha are in train_params, not instance attributes
        temperature = self.train_params.get("temperature", 3.0)
        alpha = self.train_params.get("alpha", 1.0)
        print(f"  Temperature: {temperature}, Alpha: {alpha}")

        start_time = time.time()

        #Extract data
        df = all_data["df"]
        X = all_data["X"]
        y = all_data["y"]
        input_cols = all_data["input_cols"]
        target_col = all_data["target_col"]
        id_column = all_data["id_column"]
        seq_len = all_data.get("seq_len", 24)
        
        if not _machine_unlearning_available:
            raise ImportError(
                "machine_unlearning_tool is not available. "
                "Please ensure all dependencies (including xgboost) are installed and "
                "the machine_unlearning_tool module is accessible."
            )
        
        # Convert pretrained model (use as teacher)
        pretrained_teacher = pretrained_model
        
        # eval against baseline_model (Exact RT) if set, else pretrained_teacher
        baseline_for_eval = baseline_model if baseline_model is not None else pretrained_teacher
        
        # Extract FedEraser flag from config
        use_federaser = self.train_params.get("use_federaser", False)

        #Run knowledge distillation
        result = run_knowledge_distillation(
            X=X,
            y=y,
            df=df,
            input_cols=input_cols,
            target_col=target_col,
            id_column=id_column,
            forget_ids=forget_ids,
            device=str(self.device),
            seq_len=seq_len,
            teacher_params=self.model_params,
            student_params=self.model_params,
            train_params=self.train_params,
            pretrained_teacher=pretrained_teacher,  #Still use pretrained_teacher for training
            baseline_model=baseline_for_eval,  # Use baseline (Exact Retraining) for evaluation
            baseline_original_model=baseline_original_model,
            model_type=self.model_type,
            experiment_type=self.experiment_type,
            fl_model_params=self.fl_model_params,
            test_loader=test_loader,
            test_df=test_df,
            test_input_cols=test_input_cols,
            test_target_col=test_target_col,
            # NEW FEDERASER PARAMETERS
            use_federaser=use_federaser,
            results_dir=self.results_dir,
            num_clients=self.num_clients,
            current_round=round_num
        )
        
        unlearning_time = time.time() - start_time
        
        # Extract metrics
        metrics = {}
        metrics.update(result.get("metrics_utility", {}))
        for key, val in result.get("metrics_retain", {}).items():
            metrics.setdefault(key, val)
        metrics["unlearning_time_s"] = unlearning_time
        
        #Add forget-set metrics if available
        if "metrics_forget" in result:
            metrics.update(result["metrics_forget"])
        
        #Add efficiency metrics if available
        if "efficiency_metrics" in result:
            metrics.update(result["efficiency_metrics"])
        
        return {
            "model": result["model"],
            "metrics": metrics
        }


class MFStrategy(DistillationStrategy):
    """Federaser-style distillation (a.k.a. M'F). Always enables use_federaser."""

    def __init__(
        self,
        model_type: str = "lstm",
        model_params: Optional[Dict] = None,
        train_params: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        experiment_type: Optional[str] = None,
        fl_model_params: Optional[Dict] = None,
        results_dir: Optional[str] = None,
        num_clients: Optional[int] = None
    ):
        enforced_train_params = dict(train_params or {})
        enforced_train_params["use_federaser"] = True
        enforced_train_params["mf_mode"] = True
        super().__init__(
            model_type=model_type,
            model_params=model_params,
            train_params=enforced_train_params,
            device=device,
            experiment_type=experiment_type,
            fl_model_params=fl_model_params,
            results_dir=results_dir,
            num_clients=num_clients
        )

    def unlearn(
        self,
        pretrained_model: Any,
        all_data: Dict[str, Any],
        forget_ids: List[int],
        round_num: int,
        test_loader: Optional[Any] = None,
        test_df: Optional[pd.DataFrame] = None,
        test_input_cols: Optional[List[str]] = None,
        test_target_col: Optional[str] = None,
        baseline_model: Optional[Any] = None,
        baseline_original_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        # Re-enforce flag in case caller mutated train_params
        self.train_params["use_federaser"] = True
        print("\n[MF] Running distillation with FedEraser enabled")
        return super().unlearn(
            pretrained_model=pretrained_model,
            all_data=all_data,
            forget_ids=forget_ids,
            round_num=round_num,
            test_loader=test_loader,
            test_df=test_df,
            test_input_cols=test_input_cols,
            test_target_col=test_target_col,
            baseline_model=baseline_model,
            baseline_original_model=baseline_original_model
        )


def get_strategy(strategy_name: str, **kwargs) -> UnlearningStrategy:
    """Build an unlearning strategy by name (exact_retraining/sisa/distillation/mf/federated_exact_retraining)."""
    strategies = {
        "exact_retraining": ExactRetrainingStrategy,
        "sisa": SISAStrategy,
        "distillation": DistillationStrategy,
        "mf": MFStrategy,
        "federated_exact_retraining": FederatedExactRetrainingStrategy,
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from: {list(strategies.keys())}")

    return strategies[strategy_name](**kwargs)


def collect_all_client_data(clients: Dict, experiment_type: str) -> Dict[str, Any]:
    """Pool all clients' data into one unlearning-format dict (adapter if available, else fallback)."""
    import fl_module
    from fl_module.registry import DatasetAdapterRegistry

    adapter = DatasetAdapterRegistry.get_adapter(experiment_type)
    
    if adapter:
        # Use adapter-based approach
        print(f"Using adapter for dataset type: {experiment_type}")
        return _collect_data_with_adapter(clients, adapter)
    else:
        # Fallback to hardcoded method
        print(f"Adapter not found for {experiment_type}, using fallback method")
        return _collect_data_fallback(clients, experiment_type)


def _collect_data_with_adapter(clients: Dict, adapter) -> Dict[str, Any]:
    """Collect data using adapter pattern."""
    all_samples = []
    all_labels = []
    all_client_ids = []
    
    for client_id, client in clients.items():
        #Load data using adapter
        samples, labels = adapter.load_client_data(
            client_id=client_id,
            data_dir=client.data_dir,
            sample_size=10000  #Load all available
        )
        
        # Get sequence length to determine label repetition
        seq_len = adapter.get_sequence_length()
        
        # Check original samples shape BEFORE flattening
        if samples.ndim == 3 and seq_len > 1:
            # Time series: repeat labels for each timestep
            samples_flat = adapter.flatten_samples(samples)
            labels_flat = np.repeat(labels, seq_len)
            client_ids_flat = np.repeat([client_id], len(samples_flat))
        else:
            #Non-sequential: one label per sample
            samples_flat = adapter.flatten_samples(samples)
            labels_flat = labels
            client_ids_flat = np.repeat([client_id], len(samples_flat))
        
        all_samples.append(samples_flat)
        all_labels.append(labels_flat)
        all_client_ids.append(client_ids_flat)
    
    #Validate
    if not all_samples or len(all_samples) == 0:
        raise ValueError("No data collected from clients. Cannot perform unlearning.")
    
    # Concatenate
    X = np.concatenate(all_samples, axis=0)
    y = np.concatenate(all_labels, axis=0)
    client_ids = np.concatenate(all_client_ids, axis=0)
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty dataset after concatenation. Cannot perform unlearning.")
    
    # Convert to unlearning format using adapter
    return adapter.to_unlearning_format(X, y, client_ids)


def _collect_data_fallback(clients: Dict, experiment_type: str) -> Dict[str, Any]:
    """Fallback method using hardcoded dataset-specific logic (backward compatibility)."""
    import fl_module
    
    all_samples = []
    all_labels = []
    all_client_ids = []
    
    # Collect data from all clients
    for client_id, client in clients.items():
        if experiment_type == "n_cmapss":
            #Load raw data for this client
            samples, labels = fl_module.load_ncmapss_client_data(
                client_id,
                client.data_dir,
                sample_size=10000  #Load all available
            )
            
            # Flatten sequences
            n_samples, seq_len, n_features = samples.shape
            samples_flat = samples.reshape(-1, n_features)
            
            # Repeat labels for each timestep
            labels_flat = np.repeat(labels, seq_len)
            client_ids_flat = np.repeat([client_id], len(samples_flat))
            
            all_samples.append(samples_flat)
            all_labels.append(labels_flat)
            all_client_ids.append(client_ids_flat)
        
        elif experiment_type == "mnist":
            # MNIST: flatten images
            images, labels = fl_module.load_mnist_client_data(
                client_id,
                client.data_dir,
                sample_size=10000
            )
            
            images_flat = images.reshape(len(images), -1)
            client_ids_flat = np.repeat([client_id], len(images_flat))
            
            all_samples.append(images_flat)
            all_labels.append(labels)
            all_client_ids.append(client_ids_flat)
        elif experiment_type == "cifar10":
            images, labels = fl_module.load_cifar10_client_data(
                client_id,
                client.data_dir,
                sample_size=10000
            )
            images_flat = images.reshape(len(images), -1)
            client_ids_flat = np.repeat([client_id], len(images_flat))
            all_samples.append(images_flat)
            all_labels.append(labels)
            all_client_ids.append(client_ids_flat)
        
        elif experiment_type == "tabular":
            #For tabular, data is already in the right format
            features, labels = fl_module.load_tabular_client_data(
                client_id,
                client.data_dir,
                sample_size=10000
            )
            
            #Tabular data already flat
            client_ids_flat = np.repeat([client_id], len(features))
            
            all_samples.append(features)
            all_labels.append(labels)
            all_client_ids.append(client_ids_flat)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}. No adapter registered and fallback not available.")
    
    # require data
    if not all_samples or len(all_samples) == 0:
        raise ValueError("No data collected from clients. Cannot perform unlearning.")
    
    # Concatenate all data
    X = np.concatenate(all_samples, axis=0)
    y = np.concatenate(all_labels, axis=0)
    client_ids = np.concatenate(all_client_ids, axis=0)
    
    # Validate concatenated data
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty dataset after concatenation. Cannot perform unlearning.")
    
    #Create DataFrame
    if experiment_type == "n_cmapss":
        #Create feature columns
        n_features = X.shape[1]
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["client_id"] = client_ids
        df["target"] = y
        
        input_cols = feature_cols
        target_col = "target"
        id_column = "client_id"
        seq_len = 50  # N-CMAPSS sequence length
    
    elif experiment_type == "mnist":
        # Create pixel columns
        n_pixels = X.shape[1]
        pixel_cols = [f"pixel_{i}" for i in range(n_pixels)]
        df = pd.DataFrame(X, columns=pixel_cols)
        df["client_id"] = client_ids
        df["target"] = y
        
        input_cols = pixel_cols
        target_col = "target"
        id_column = "client_id"
        seq_len = 1  # MNIST doesn't use sequences
    
    elif experiment_type == "tabular":
        #Create feature columns
        n_features = X.shape[1]
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df["client_id"] = client_ids
        df["target"] = y
        
        input_cols = feature_cols
        target_col = "target"
        id_column = "client_id"
        seq_len = 1  #Tabular data doesn't use sequences
    
    return {
        "df": df,
        "X": X,
        "y": y,
        "input_cols": input_cols,
        "target_col": target_col,
        "id_column": id_column,
        "seq_len": seq_len
    }


class FederatedExactRetrainingStrategy(UnlearningStrategy):
    """Federated Exact Retraining - True FL-native unlearning (golden standard).

    This strategy implements the paper-equivalent exact retraining for federated learning:
    1. Replays the ENTIRE FL training process from scratch
    2. Excludes forget_clients from ALL rounds
    3. Performs client-local training + FedAvg aggregation per round
    4. Reproduces the same FL dynamics (rounds, local epochs, aggregation)

    This is the TRUE golden standard for FL unlearning, unlike centralized retraining
    which trains on pooled data without FL rounds/aggregation.

    References:
        - FedEraser (Liu et al., 2021): "Machine Unlearning in Federated Learning"
        - KNOT (Halimi et al., 2022): "Federated Unlearning: How to Efficiently Erase Users"
        - Subspace FL Unlearning (Gao et al., 2024)
    """

    def __init__(self, *args, orchestrator_ref=None, **kwargs):
        # orchestrator_ref: replays the FL rounds
        super().__init__(*args, **kwargs)
        self.orchestrator_ref = orchestrator_ref

    def unlearn(
        self,
        pretrained_model: Any,
        all_data: Dict[str, Any],
        forget_ids: List[int],
        round_num: int,
        test_loader: Optional[Any] = None,
        test_df: Optional[pd.DataFrame] = None,
        test_input_cols: Optional[List[str]] = None,
        test_target_col: Optional[str] = None,
        baseline_model: Optional[Any] = None,
        baseline_original_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        r"""Replay the whole FL process excluding the forget clients (true FL golden standard):

        for t in 1..T:
            C_t = sample_clients(all_clients \\ forget_clients)
            for c in C_t:
                Δw_c = local_train(M_t, D_c, local_epochs)
            M_{t+1} = FedAvg([Δw_c for c in C_t])
        """
        import copy
        from torch.utils.data import DataLoader, TensorDataset

        print(f"\nFederated exact retraining: replaying rounds 1..{round_num} "
              f"without clients {forget_ids} (full FedAvg loop, golden standard).")

        start_time = time.time()

        # clients_ref is required to replay the FL loop
        if not hasattr(self, 'clients_ref') or self.clients_ref is None:
            raise ValueError("FederatedExactRetrainingStrategy requires clients_ref to be set")

        all_clients = self.clients_ref
        retain_client_ids = [cid for cid in all_clients.keys() if cid not in forget_ids]

        print(f"Total clients: {len(all_clients)}")
        print(f"Forget clients: {forget_ids}")
        print(f"Retain clients: {retain_client_ids}")
        print(f"FL rounds to replay: {round_num}\n")

        # Start from a freshly initialised model with the same architecture.
        #prefer fl_module.create_model for identical init; else deepcopy + Xavier reset
        if self.experiment_type:
            try:
                from fl_module import create_model as create_fl_model
                unlearned_model = create_fl_model(
                    experiment_type=self.experiment_type,
                    **self.fl_model_params
                )
                unlearned_model = unlearned_model.to(self.device)
                print(f"Created fresh {self.experiment_type} model from fl_module")
            except Exception as e:
                print(f"Warning: Could not create FL model: {e}. Using deepcopy.")
                unlearned_model = copy.deepcopy(pretrained_model)
                #Reset weights
                for module in unlearned_model.modules():
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
                    elif isinstance(module, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
        else:
            unlearned_model = copy.deepcopy(pretrained_model)
            # Reset weights
            for module in unlearned_model.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

        unlearned_model.train()

        # Replay the FL loop over the retain clients. training_time_s only
        # counts actual client-side training - that's the fair measure of
        #unlearning cost, excluding aggregation and evaluation.
        fl_history = []
        training_time_s = 0.0

        for r in range(1, round_num + 1):
            print(f"\n--- Replaying Round {r}/{round_num} ---")
            round_start = time.time()

            #snapshot of current global model; every client starts here
            global_state = copy.deepcopy(unlearned_model.state_dict())

            client_updates = []
            client_data_sizes = []

            for client_id in retain_client_ids:
                client = all_clients[client_id]

                # Load client's local data
                # Create a temporary model for this client
                client_model = copy.deepcopy(unlearned_model)
                client_model.load_state_dict(global_state)
                client_model.train()

                # Train locally
                optimizer = torch.optim.Adam(
                    client_model.parameters(),
                    lr=self.train_params.get("learning_rate", 0.001)
                )

                #Determine loss function (plug-and-play: use adapter if available)
                try:
                    from fl_module.registry import DatasetAdapterRegistry
                    adapter = DatasetAdapterRegistry.get_adapter(self.experiment_type)
                    is_cls = adapter.is_classification() if adapter is not None else self.experiment_type in ["mnist", "cifar10", "tabular"]
                except Exception:
                    is_cls = self.experiment_type in ["mnist", "cifar10", "tabular"]
                criterion = torch.nn.CrossEntropyLoss() if is_cls else torch.nn.MSELoss()

                #Train for local_epochs (only this counts as training_time_s)
                local_epochs = self.train_params.get("local_epochs", 5)
                total_samples = 0
                client_train_t0 = time.time()

                for epoch in range(local_epochs):
                    epoch_loss = 0.0
                    n_batches = 0

                    for batch_X, batch_y in client.train_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        optimizer.zero_grad()
                        outputs = client_model(batch_X)

                        # Handle output shape
                        if criterion.__class__.__name__ == 'CrossEntropyLoss':
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs.squeeze(), batch_y.float())

                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        n_batches += 1
                        total_samples += len(batch_X)

                    avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
                    if epoch == 0:  # Only print first epoch to reduce spam
                        print(f"  Client {client_id} | Epoch {epoch+1}/{local_epochs} | Loss: {avg_loss:.4f}")

                training_time_s += time.time() - client_train_t0

                # Store client update (model state dict)
                client_updates.append(client_model.state_dict())
                client_data_sizes.append(total_samples // local_epochs)  #Approximate

            #FedAvg: weighted average by each client's dataset size.
            print(f"\n  Aggregating {len(client_updates)} client models via FedAvg...")

            if len(client_updates) == 0:
                print("  Warning: No client updates to aggregate. Keeping current model.")
            else:
                # Weighted average by data size
                total_data = sum(client_data_sizes)
                aggregated_state = {}

                for key in client_updates[0].keys():
                    aggregated_state[key] = sum(
                        client_updates[i][key] * (client_data_sizes[i] / total_data)
                        for i in range(len(client_updates))
                    )

                unlearned_model.load_state_dict(aggregated_state)

            round_time = time.time() - round_start
            print(f"  Round {r} completed in {round_time:.2f}s")

            fl_history.append({
                "round": r,
                "num_clients": len(retain_client_ids),
                "round_time": round_time
            })

        total_time = time.time() - start_time
        print(f"\nFederated Exact Retraining completed in {total_time:.2f}s")
        print(f"Average time per round: {total_time / round_num:.2f}s\n")

        # Eval on the unlearned model.
        unlearned_model.eval()

        from machine_unlearning_tool.evaluation import evaluate_model_universal

        metrics = {}
        if test_loader is not None:
            try:
                from fl_module.registry import DatasetAdapterRegistry
                adapter = DatasetAdapterRegistry.get_adapter(self.experiment_type)
                is_classification = adapter.is_classification() if adapter is not None else self.experiment_type in ["mnist", "cifar10", "tabular"]
            except Exception:
                is_classification = self.experiment_type in ["mnist", "cifar10", "tabular"]
            test_metrics = evaluate_model_universal(
                unlearned_model,
                loader=test_loader,
                device=self.device,
                is_classification=is_classification
            )
            metrics.update(test_metrics)

            if is_classification:
                print(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
            else:
                print(f"Test MAE: {test_metrics.get('mae', 0):.4f}")

        metrics["unlearning_time_s"] = total_time
        metrics["training_time_s"] = training_time_s

        # ── Forget-set evaluation ─────────────────────────────────────────────
        #Mirrors the evaluation done in run_exact_retraining so that
        #federated_exact_retraining reports forget_accuracy, MIA, JS divergence, etc.
        metrics_forget = {}
        original_model_for_eval = baseline_original_model if baseline_original_model is not None else pretrained_model
        try:
            if (original_model_for_eval is not None
                    and is_pytorch_model(original_model_for_eval)
                    and is_pytorch_model(unlearned_model)):

                from machine_unlearning_tool.evaluation import (
                    evaluate_model_universal,
                    compute_per_sample_losses,
                    simple_mia,
                    compute_confidence_metrics,
                    compute_js_divergence,
                )
                from torch.utils.data import TensorDataset, DataLoader as _DL

                # Build forget loader from all_data + forget_ids
                df = all_data["df"]
                X = all_data["X"]
                y = all_data["y"]
                id_col = all_data["id_column"]
                seq_len = all_data.get("seq_len", 1)

                forget_mask = df[id_col].isin(forget_ids)
                X_forget = X[forget_mask.values]
                y_forget = y[forget_mask.values]

                if len(X_forget) > 0:
                    X_t = torch.from_numpy(X_forget).float()
                    if self.experiment_type == "mnist" and X_t.dim() == 2:
                        X_t = X_t.view(-1, 1, 28, 28)
                    elif self.experiment_type == "cifar10" and X_t.dim() == 2:
                        X_t = X_t.view(-1, 3, 32, 32)
                    y_t = torch.from_numpy(y_forget).long() if is_classification else torch.from_numpy(y_forget).float()
                    loader_forget = _DL(TensorDataset(X_t, y_t), batch_size=64, shuffle=False)

                    original_model_for_eval.eval()
                    unlearned_model.eval()

                    m_orig = evaluate_model_universal(original_model_for_eval, loader=loader_forget,
                                                      device=self.device, is_classification=is_classification)
                    m_unl  = evaluate_model_universal(unlearned_model, loader=loader_forget,
                                                      device=self.device, is_classification=is_classification)

                    if is_classification:
                        acc_orig = m_orig.get("accuracy", 0.0)
                        acc_unl  = m_unl.get("accuracy", 0.0)
                        metrics_forget.update({
                            "forget_accuracy_original": acc_orig,
                            "forget_accuracy_unlearned": acc_unl,
                            "unlearning_score": acc_orig - acc_unl,
                        })
                    else:
                        metrics_forget.update({
                            "forget_rmse_original": m_orig.get("rmse", float("inf")),
                            "forget_rmse_unlearned": m_unl.get("rmse", float("inf")),
                        })

                    # MIA
                    if test_loader is not None:
                        try:
                            import numpy as np
                            tl_orig = compute_per_sample_losses(original_model_for_eval, test_loader,
                                                                self.device, is_classification=is_classification)
                            tl_unl  = compute_per_sample_losses(unlearned_model, test_loader,
                                                                self.device, is_classification=is_classification)
                            fl_orig = compute_per_sample_losses(original_model_for_eval, loader_forget,
                                                                self.device, is_classification=is_classification)
                            fl_unl  = compute_per_sample_losses(unlearned_model, loader_forget,
                                                                self.device, is_classification=is_classification)

                            n = min(len(tl_orig), len(fl_orig))
                            tl_orig, tl_unl = tl_orig[:n], tl_unl[:n]
                            fl_orig, fl_unl = fl_orig[:n], fl_unl[:n]

                            mia_orig = simple_mia(np.concatenate([tl_orig, fl_orig]),
                                                  np.concatenate([np.zeros(n), np.ones(n)]))
                            mia_unl  = simple_mia(np.concatenate([tl_unl,  fl_unl]),
                                                  np.concatenate([np.zeros(n), np.ones(n)]))
                            metrics_forget.update({
                                "mia_accuracy_original":  mia_orig["mia_accuracy"],
                                "mia_accuracy_unlearned": mia_unl["mia_accuracy"],
                                "mia_improvement": mia_orig["mia_accuracy"] - mia_unl["mia_accuracy"],
                            })
                        except Exception as mia_err:
                            print(f"  [fed_exact] MIA calculation skipped: {mia_err}")

                    # Confidence / entropy / JS divergence on forget set
                    try:
                        conf = compute_confidence_metrics(unlearned_model, loader_forget, self.device)
                        metrics_forget.update({
                            "forget_confidence_mean_unlearned": conf.get("confidence_mean"),
                            "forget_entropy_mean_unlearned":    conf.get("entropy_mean"),
                        })
                        conf_orig = compute_confidence_metrics(original_model_for_eval, loader_forget, self.device)
                        metrics_forget.update({
                            "forget_confidence_mean_original": conf_orig.get("confidence_mean"),
                            "forget_entropy_mean_original":    conf_orig.get("entropy_mean"),
                        })
                        js = compute_js_divergence(original_model_for_eval, unlearned_model, loader_forget, self.device)
                        metrics_forget.update(js)
                    except Exception as conf_err:
                        print(f"  [fed_exact] Confidence/JS metrics skipped: {conf_err}")

        except Exception as eval_err:
            print(f"  [fed_exact] Forget evaluation skipped: {eval_err}")

        metrics.update(metrics_forget)
        metrics["utility_accuracy_test"] = metrics.get("accuracy")

        return {
            "model": unlearned_model,
            "unlearned_model": unlearned_model,
            "ensemble": unlearned_model,
            "metrics": metrics,
            "metrics_utility": metrics,
            "metrics_retain": metrics,
            "metrics_forget": metrics_forget,
            "timing": {
                "total_time": total_time,
                "avg_time_per_round": total_time / round_num,
                "training_time_s": training_time_s
            },
            "fl_history": fl_history,
            "retrain_fraction": 1.0,
            "method": "federated_exact_retraining"
        }
