"""Branch registry for managing unlearning checkpoints and branches."""

import os
import json
import pickle
import torch
from datetime import datetime
from typing import Dict, Optional, List, Any, Union


class BranchRegistry:
    """Manages checkpoints and branches for unlearning operations.
    
    Each policy change (e.g., client exclusion) creates a branch point where
    multiple unlearning strategies can be evaluated. This registry tracks:
    - Pre-unlearning checkpoints (shared across all strategies)
    - Post-unlearning checkpoints (one per strategy/branch)
    - Metrics and metadata for each branch
    """
    
    def __init__(self, results_dir: str, round_num: int):
        self.results_dir = results_dir
        self.round_num = round_num

        self.structure = self._get_structure_config()

        round_dir = os.path.join(
            self.results_dir,
            self.structure["round_template"].format(round=round_num)
        )
        self.unlearning_dir = os.path.join(round_dir, "unlearning")
        self.pre_unlearning_dir = os.path.join(self.unlearning_dir, "pre_unlearning")
        self.branches_dir = os.path.join(self.unlearning_dir, "branches")
        
        os.makedirs(self.unlearning_dir, exist_ok=True)
        os.makedirs(self.pre_unlearning_dir, exist_ok=True)
        os.makedirs(self.branches_dir, exist_ok=True)
    
    def _get_structure_config(self) -> Dict:
        """Get directory structure configuration."""
        default_structure = {
            "round_template": "model_storage/round_{round}",
        }
        
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(self.results_dir)),
            "mock_etcd/configuration.json"
        )
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "results" in config and "structure" in config["results"]:
                        loaded = config["results"]["structure"]
                        if "round_template" in loaded:
                            default_structure["round_template"] = loaded["round_template"]
        except Exception as e:
            print(f"Warning: Could not load structure config: {e}")
        
        return default_structure
    
    def save_pre_unlearning_checkpoint(
        self,
        model_state_dict: Union[Dict, Any],
        forget_ids: List[int],
        metadata: Optional[Dict] = None
    ) -> str:
        """Save the shared pre-unlearning checkpoint (PyTorch state dict or sklearn model)."""
        is_sklearn = False
        actual_model = model_state_dict
        
        if isinstance(model_state_dict, dict):
            if model_state_dict.get("type") == "sklearn":
                is_sklearn = True
                actual_model = model_state_dict.get("model")
            elif model_state_dict.get("is_sklearn", False):
                is_sklearn = True
                actual_model = model_state_dict.get("model", model_state_dict)
        else:
            #Check if it's a sklearn model directly
            try:
                from sklearn.base import BaseEstimator
                if isinstance(model_state_dict, BaseEstimator):
                    is_sklearn = True
                    actual_model = model_state_dict
            except ImportError:
                pass
        
        #Save model based on type
        if is_sklearn:
            model_path = os.path.join(self.pre_unlearning_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(actual_model, f)
        else:
            model_path = os.path.join(self.pre_unlearning_dir, "model.pt")
            torch.save(model_state_dict, model_path)
        
        # Save metadata
        checkpoint_metadata = {
            "round": self.round_num,
            "forget_ids": forget_ids,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_type": "pre_unlearning",
            "is_sklearn": is_sklearn
        }
        
        if metadata:
            checkpoint_metadata.update(metadata)
        
        metadata_path = os.path.join(self.pre_unlearning_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        
        print(f"Saved pre-unlearning checkpoint to {self.pre_unlearning_dir}")
        return self.pre_unlearning_dir
    
    def save_branch_checkpoint(
        self,
        strategy_name: str,
        model_state_dict: Union[Dict, Any],
        metrics: Dict[str, float],
        metadata: Optional[Dict] = None
    ) -> str:
        """Save a strategy's post-unlearning checkpoint plus its metrics/metadata."""
        branch_dir = os.path.join(self.branches_dir, strategy_name)
        os.makedirs(branch_dir, exist_ok=True)

        is_sklearn = False
        actual_model = model_state_dict
        
        if isinstance(model_state_dict, dict):
            if model_state_dict.get("type") == "sklearn":
                is_sklearn = True
                actual_model = model_state_dict.get("model")
                if actual_model is None:
                    raise ValueError(f"Sklearn model not found in model_state_dict for strategy {strategy_name}")
        else:
            # Check if it's a sklearn model directly
            try:
                from sklearn.base import BaseEstimator
                if isinstance(model_state_dict, BaseEstimator):
                    is_sklearn = True
                    actual_model = model_state_dict
            except ImportError:
                pass
        
        # Save model based on type
        if is_sklearn:
            model_path = os.path.join(branch_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(actual_model, f)
        else:
            model_path = os.path.join(branch_dir, "model.pt")
            torch.save(model_state_dict, model_path)
        
        #Save metrics
        metrics_path = os.path.join(branch_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        #Save metadata
        branch_metadata = {
            "round": self.round_num,
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_type": "post_unlearning",
            "metrics": metrics,
            "is_sklearn": is_sklearn
        }
        
        if metadata:
            branch_metadata.update(metadata)
        
        metadata_path = os.path.join(branch_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(branch_metadata, f, indent=2)
        
        print(f"Saved branch checkpoint for '{strategy_name}' to {branch_dir}")
        return branch_dir
    
    def load_pre_unlearning_checkpoint(self) -> Optional[Union[Dict, Any]]:
        """Load the pre-unlearning checkpoint (None if absent)."""
        metadata_path = os.path.join(self.pre_unlearning_dir, "metadata.json")
        is_sklearn = False
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    is_sklearn = metadata.get("is_sklearn", False)
            except:
                pass
        
        if is_sklearn:
            model_path = os.path.join(self.pre_unlearning_dir, "model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        else:
            model_path = os.path.join(self.pre_unlearning_dir, "model.pt")
            if os.path.exists(model_path):
                return torch.load(model_path, map_location='cpu')
        return None
    
    def load_branch_checkpoint(self, strategy_name: str) -> Optional[Union[Dict, Any]]:
        """Load a strategy branch's post-unlearning checkpoint (None if absent)."""
        branch_dir = os.path.join(self.branches_dir, strategy_name)

        metadata_path = os.path.join(branch_dir, "metadata.json")
        is_sklearn = False
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    is_sklearn = metadata.get("is_sklearn", False)
            except Exception:
                pass
        
        # Load model based on type
        if is_sklearn:
            model_path = os.path.join(branch_dir, "model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        else:
            model_path = os.path.join(branch_dir, "model.pt")
            if os.path.exists(model_path):
                return torch.load(model_path, map_location='cpu')
        
        return None
    
    def get_branch_metrics(self, strategy_name: str) -> Optional[Dict]:
        """Metrics dict for a branch (None if absent)."""
        branch_dir = os.path.join(self.branches_dir, strategy_name)
        metrics_path = os.path.join(branch_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return None
    
    def save_comparison(self, comparison_data: Dict[str, Any]):
        """Save the cross-branch comparison as JSON."""
        comparison_path = os.path.join(self.unlearning_dir, "comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"Saved branch comparison to {comparison_path}")
    
    def list_branches(self) -> List[str]:
        """Strategy names that have a saved checkpoint."""
        if not os.path.exists(self.branches_dir):
            return []

        branches = []
        for item in os.listdir(self.branches_dir):
            branch_path = os.path.join(self.branches_dir, item)
            if os.path.isdir(branch_path):
                if os.path.exists(os.path.join(branch_path, "model.pt")) or \
                   os.path.exists(os.path.join(branch_path, "model.pkl")):
                    branches.append(item)
        return branches
    
    def find_existing_branch(
        self,
        forget_ids: List[int],
        strategy_name: str,
        model_type: str,
        max_search_rounds: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Search previous rounds for a branch with the same policy (forget_ids+strategy+model_type) to reuse."""
        # policy signature: sorted forget_ids + strategy + model_type
        policy_signature = {
            "forget_ids": sorted(forget_ids),
            "strategy": strategy_name,
            "model_type": model_type
        }
        
        # Search backwards through previous rounds
        for prev_round in range(self.round_num - 1, max(0, self.round_num - max_search_rounds - 1), -1):
            prev_round_dir = os.path.join(
                self.results_dir,
                self.structure["round_template"].format(round=prev_round)
            )
            prev_unlearning_dir = os.path.join(prev_round_dir, "unlearning")
            prev_branches_dir = os.path.join(prev_unlearning_dir, "branches")
            
            if not os.path.exists(prev_branches_dir):
                continue
            
            #Check if this strategy branch exists in previous round
            prev_branch_dir = os.path.join(prev_branches_dir, strategy_name)
            if not os.path.exists(prev_branch_dir):
                continue
            
            #Load metadata to check policy signature
            prev_metadata_path = os.path.join(prev_branch_dir, "metadata.json")
            if not os.path.exists(prev_metadata_path):
                continue
            
            try:
                with open(prev_metadata_path, 'r') as f:
                    prev_metadata = json.load(f)
                
                # Check if forget_ids match (sorted)
                prev_forget_ids = prev_metadata.get("forget_ids", [])
                if sorted(prev_forget_ids) != policy_signature["forget_ids"]:
                    continue
                
                # Check if model type matches (if stored)
                prev_model_type = prev_metadata.get("model_type")
                if prev_model_type and prev_model_type != model_type:
                    continue
                
                # matching branch found
                #Check if it's sklearn or PyTorch model
                is_sklearn = prev_metadata.get("is_sklearn", False)
                if is_sklearn:
                    prev_model_path = os.path.join(prev_branch_dir, "model.pkl")
                else:
                    prev_model_path = os.path.join(prev_branch_dir, "model.pt")
                
                if os.path.exists(prev_model_path):
                    return {
                        "round": prev_round,
                        "path": prev_model_path,
                        "metadata": prev_metadata,
                        "branch_dir": prev_branch_dir,
                        "is_sklearn": is_sklearn
                    }
            except Exception as e:
                print(f"Warning: Could not check branch in round {prev_round}: {e}")
                continue
        
        return None

