from typing import Dict, Iterable, List, Tuple, Union, Optional
import collections
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from .data_utils import (
    ArraySequenceDataset,
    drop_rows,
    exclude_ids,
    extract_features_targets,
    filter_by_id,
    split_into_slices,
)
from .evaluation import (
    evaluate_model_universal,
    rmse,
    mae,
    r2,
    calculate_behavioral_distance,
    compute_per_sample_losses,
    compute_per_sample_losses_sklearn,
    simple_mia,
    compute_activation_distance,
    compute_confidence_metrics,
    compute_js_divergence
)
from .model_utils import create_model, is_pytorch_model, is_sklearn_model
from .training import create_loader, train_model_universal, train_with_soft_labels


def _validate_forget_ids(df, id_column: str, forget_ids: Iterable) -> Dict:
    """Check which forget_ids actually occur in the dataset."""
    unique_ids = set(df[id_column].unique())
    forget_set = set(forget_ids)

    found = forget_set.intersection(unique_ids)
    missing = forget_set - unique_ids

    if missing:
        print(f"Warning: {len(missing)} forget_ids not found in dataset: {sorted(list(missing))}")

    n_samples = len(df[df[id_column].isin(found)])

    return {
        "found_ids": list(found),
        "missing_ids": list(missing),
        "n_samples_to_forget": n_samples,
        "n_ids_to_forget": len(found)
    }


def _extract_data_from_loader(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Extract X and y arrays from DataLoader for sklearn models."""
    X_list = []
    y_list = []
    for xb, yb in loader:
        X_list.append(xb.numpy() if isinstance(xb, torch.Tensor) else xb)
        y_list.append(yb.numpy() if isinstance(yb, torch.Tensor) else yb)
    X = np.concatenate(X_list, axis=0) if X_list else np.array([], dtype=np.float32)
    y = np.concatenate(y_list, axis=0) if y_list else np.array([], dtype=np.float32)
    return X, y


def _create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences from 2D arrays for tree-based models."""
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def _determine_is_classification(
    experiment_type: Optional[str] = None,
    fl_model_params: Optional[Dict] = None,
    model_params: Optional[Dict] = None,
    pretrained_model=None,
    adapter=None,
) -> bool:
    """Decide whether the task is classification or regression."""
    if adapter is not None and hasattr(adapter, "is_classification") and callable(getattr(adapter, "is_classification")):
        return adapter.is_classification()
    if experiment_type in ("mnist", "cifar10", "adult"):
        return True

    if experiment_type == "n_cmapss":
        return False

    if experiment_type in ("tabular", "adult"):
        output_dim = None

        if fl_model_params and "output_dim" in fl_model_params:
            output_dim = fl_model_params["output_dim"]
        elif model_params and "output_dim" in model_params:
            output_dim = model_params["output_dim"]
        elif pretrained_model is not None:
            if hasattr(pretrained_model, "output_dim"):
                output_dim = pretrained_model.output_dim
            elif is_pytorch_model(pretrained_model):
                # last Linear layer's out_features = num classes
                try:
                    for module in reversed(list(pretrained_model.modules())):
                        if isinstance(module, torch.nn.Linear):
                            output_dim = module.out_features
                            break
                except:
                    pass

        # >1 outputs = classification, 1 = regression
        if output_dim is not None:
            return output_dim > 1

        return True

    return False


def _standardize_utility_metrics(metrics: Dict[str, float], dataset_label: str = "test") -> Dict[str, float]:
    """Expose explicit utility_* names while keeping original keys."""
    if not metrics:
        return {}

    label = dataset_label or "test"
    standardized: Dict[str, float] = {}

    if "accuracy" in metrics:
        standardized[f"utility_accuracy_{label}"] = float(metrics["accuracy"])
    if "precision" in metrics:
        standardized[f"utility_precision_{label}"] = float(metrics["precision"])
    if "recall" in metrics:
        standardized[f"utility_recall_{label}"] = float(metrics["recall"])
    if "f1" in metrics:
        standardized[f"utility_f1_{label}"] = float(metrics["f1"])
    if "rmse" in metrics:
        standardized[f"utility_rmse_{label}"] = float(metrics["rmse"])
    if "mae" in metrics:
        standardized[f"utility_mae_{label}"] = float(metrics["mae"])
    if "r2" in metrics:
        standardized[f"utility_r2_{label}"] = float(metrics["r2"])
    if "test_loss" in metrics:
        standardized[f"utility_loss_{label}"] = float(metrics["test_loss"])

    return {**metrics, **standardized}


def _evaluate_forget_set(
    original_model,
    unlearned_model,
    df,
    input_cols: List[str],
    target_col: str,
    id_column: str,
    forget_ids: Iterable,
    seq_len: int,
    device,
    train_params: Dict,
    experiment_type: str = None,
    is_classification: bool = False,
    test_loader: Optional[DataLoader] = None,
    test_df: Optional[pd.DataFrame] = None,
    test_input_cols: Optional[List[str]] = None,
    test_target_col: Optional[str] = None
) -> Dict[str, float]:
    """Evalueert origineel en unlearned model op forget set, met optionele MIA."""
    #rebuild original model from state_dict (unlearned arch) for eval
    if not is_pytorch_model(original_model) and isinstance(original_model, (dict, collections.OrderedDict)):
        if is_pytorch_model(unlearned_model):
            import copy
            rebuilt = copy.deepcopy(unlearned_model)
            try:
                rebuilt.load_state_dict(original_model)
                rebuilt.eval()
                original_model = rebuilt
            except RuntimeError as e:
                print(f"Warning: Could not load state dict into rebuilt model: {e}")
                import traceback
                traceback.print_exc()
                original_model = None
        else:
            original_model = None

    forget_df = filter_by_id(df, id_column, forget_ids)
    if len(forget_df) == 0:
        return {}
    
    X_forget, y_forget = extract_features_targets(forget_df, input_cols, target_col)
    
    if is_pytorch_model(unlearned_model):
        if seq_len == 1 and experiment_type in ["tabular", "adult", "mnist", "cifar10"]:
            X_forget_tensor = torch.from_numpy(X_forget).float()
            if experiment_type == "mnist" and X_forget_tensor.dim() == 2:
                X_forget_tensor = X_forget_tensor.view(-1, 1, 28, 28)
            elif experiment_type == "cifar10" and X_forget_tensor.dim() == 2:
                X_forget_tensor = X_forget_tensor.view(-1, 3, 32, 32)
            y_forget_tensor = torch.from_numpy(y_forget).long() if is_classification else torch.from_numpy(y_forget).float()
            ds_forget = TensorDataset(X_forget_tensor, y_forget_tensor)
        else:
            ds_forget = ArraySequenceDataset(X_forget, y_forget, seq_len=seq_len)
        loader_forget = create_loader(ds_forget, batch_size=train_params.get("batch_size", 64))
        
        unlearned_model.eval()
        metrics_forget_unlearned = evaluate_model_universal(
            unlearned_model, loader=loader_forget, device=device, is_classification=is_classification
        )
    else:
        #sklearn
        if seq_len == 1:
            X_forget_flat = X_forget
        else:
            X_forget_seq, y_forget_seq = _create_sequences(X_forget, y_forget, seq_len)
            X_forget_flat = X_forget_seq.reshape(X_forget_seq.shape[0], -1) if X_forget_seq.ndim == 3 else X_forget_seq
        
        y_pred_unlearned = unlearned_model.predict(X_forget_flat)
        if is_classification:
            y_pred_classes = np.argmax(y_pred_unlearned, axis=1) if y_pred_unlearned.ndim > 1 else y_pred_unlearned
            accuracy_unlearned = np.mean(y_pred_classes == y_forget)
            metrics_forget_unlearned = {"accuracy": float(accuracy_unlearned)}
        else:
            metrics_forget_unlearned = {
                "rmse": rmse(y_forget, y_pred_unlearned),
                "mae": mae(y_forget, y_pred_unlearned),
                "r2": r2(y_forget, y_pred_unlearned)
            }
    
    if original_model is None:
        metrics_forget_original = {}
    elif is_pytorch_model(original_model):
        if seq_len == 1 and experiment_type in ["tabular", "adult", "mnist", "cifar10"]:
            X_forget_tensor = torch.from_numpy(X_forget).float()
            if experiment_type == "mnist" and X_forget_tensor.dim() == 2:
                X_forget_tensor = X_forget_tensor.view(-1, 1, 28, 28)
            elif experiment_type == "cifar10" and X_forget_tensor.dim() == 2:
                X_forget_tensor = X_forget_tensor.view(-1, 3, 32, 32)
            y_forget_tensor = torch.from_numpy(y_forget).long() if is_classification else torch.from_numpy(y_forget).float()
            ds_forget = TensorDataset(X_forget_tensor, y_forget_tensor)
        else:
            ds_forget = ArraySequenceDataset(X_forget, y_forget, seq_len=seq_len)
        loader_forget = create_loader(ds_forget, batch_size=train_params.get("batch_size", 64))
        
        original_model.eval()
        metrics_forget_original = evaluate_model_universal(
            original_model, loader=loader_forget, device=device, is_classification=is_classification
        )
    else:
        # sklearn
        if seq_len == 1:
            X_forget_flat = X_forget
        else:
            X_forget_seq, y_forget_seq = _create_sequences(X_forget, y_forget, seq_len)
            X_forget_flat = X_forget_seq.reshape(X_forget_seq.shape[0], -1) if X_forget_seq.ndim == 3 else X_forget_seq
        
        y_pred_original = original_model.predict(X_forget_flat)
        if is_classification:
            y_pred_classes = np.argmax(y_pred_original, axis=1) if y_pred_original.ndim > 1 else y_pred_original
            accuracy_original = np.mean(y_pred_classes == y_forget)
            metrics_forget_original = {"accuracy": float(accuracy_original)}
        else:
            metrics_forget_original = {
                "rmse": rmse(y_forget, y_pred_original),
                "mae": mae(y_forget, y_pred_original),
                "r2": r2(y_forget, y_pred_original)
            }
    
    if is_classification:
        acc_orig = metrics_forget_original.get("accuracy", 0.0)
        acc_unl = metrics_forget_unlearned.get("accuracy", 0.0)
        unlearning_score = acc_orig - acc_unl
        result = {
            "forget_accuracy_original": acc_orig,
            "forget_accuracy_unlearned": acc_unl,
            "unlearning_score": unlearning_score
        }
    else:
        rmse_orig = metrics_forget_original.get("rmse", float("inf"))
        rmse_unl = metrics_forget_unlearned.get("rmse", float("inf"))
        unlearning_score = rmse_unl - rmse_orig
        result = {
            "forget_rmse_original": rmse_orig,
            "forget_rmse_unlearned": rmse_unl,
            "unlearning_score": unlearning_score
        }
    
    mia_metrics = {}

    if test_loader is not None and is_pytorch_model(original_model) and is_pytorch_model(unlearned_model):
        try:
            test_losses_orig = compute_per_sample_losses(
                original_model, test_loader, device, is_classification=is_classification
            )
            test_losses_unl = compute_per_sample_losses(
                unlearned_model, test_loader, device, is_classification=is_classification
            )

            forget_losses_orig = compute_per_sample_losses(
                original_model, loader_forget, device, is_classification=is_classification
            )
            forget_losses_unl = compute_per_sample_losses(
                unlearned_model, loader_forget, device, is_classification=is_classification
            )

            # balance test/forget sizes so the MIA classifier isn't biased
            min_len = min(len(test_losses_orig), len(forget_losses_orig))
            if len(test_losses_orig) > min_len:
                indices = np.random.choice(len(test_losses_orig), min_len, replace=False)
                test_losses_orig = test_losses_orig[indices]
                test_losses_unl = test_losses_unl[indices]
            if len(forget_losses_orig) > min_len:
                indices = np.random.choice(len(forget_losses_orig), min_len, replace=False)
                forget_losses_orig = forget_losses_orig[indices]
                forget_losses_unl = forget_losses_unl[indices]

            samples_mia_orig = np.concatenate([test_losses_orig, forget_losses_orig])
            labels_mia_orig = np.concatenate([
                np.zeros(len(test_losses_orig)),
                np.ones(len(forget_losses_orig))
            ])
            mia_result_orig = simple_mia(samples_mia_orig, labels_mia_orig)

            samples_mia_unl = np.concatenate([test_losses_unl, forget_losses_unl])
            labels_mia_unl = np.concatenate([
                np.zeros(len(test_losses_unl)),
                np.ones(len(forget_losses_unl))
            ])
            mia_result_unl = simple_mia(samples_mia_unl, labels_mia_unl)

            mia_metrics = {
                "mia_accuracy_original": mia_result_orig["mia_accuracy"],
                "mia_accuracy_unlearned": mia_result_unl["mia_accuracy"],
                "mia_improvement": mia_result_orig["mia_accuracy"] - mia_result_unl["mia_accuracy"]
            }
        except Exception as e:
            print(f"Warning: MIA calculation failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif test_df is not None and test_input_cols is not None and test_target_col is not None:
        try:
            X_test, y_test = extract_features_targets(test_df, test_input_cols, test_target_col)

            test_losses_orig = compute_per_sample_losses_sklearn(
                original_model, X_test, y_test, is_classification=is_classification
            )
            test_losses_unl = compute_per_sample_losses_sklearn(
                unlearned_model, X_test, y_test, is_classification=is_classification
            )

            if seq_len == 1:
                X_forget_flat = X_forget
            else:
                X_forget_seq, y_forget_seq = _create_sequences(X_forget, y_forget, seq_len)
                X_forget_flat = X_forget_seq.reshape(X_forget_seq.shape[0], -1) if X_forget_seq.ndim == 3 else X_forget_seq

            forget_losses_orig = compute_per_sample_losses_sklearn(
                original_model, X_forget_flat, y_forget, is_classification=is_classification
            )
            forget_losses_unl = compute_per_sample_losses_sklearn(
                unlearned_model, X_forget_flat, y_forget, is_classification=is_classification
            )

            # balance test/forget sizes so the MIA classifier isn't biased
            min_len = min(len(test_losses_orig), len(forget_losses_orig))
            if len(test_losses_orig) > min_len:
                indices = np.random.choice(len(test_losses_orig), min_len, replace=False)
                test_losses_orig = test_losses_orig[indices]
                test_losses_unl = test_losses_unl[indices]
            if len(forget_losses_orig) > min_len:
                indices = np.random.choice(len(forget_losses_orig), min_len, replace=False)
                forget_losses_orig = forget_losses_orig[indices]
                forget_losses_unl = forget_losses_unl[indices]

            samples_mia_orig = np.concatenate([test_losses_orig, forget_losses_orig])
            labels_mia_orig = np.concatenate([
                np.zeros(len(test_losses_orig)),
                np.ones(len(forget_losses_orig))
            ])
            mia_result_orig = simple_mia(samples_mia_orig, labels_mia_orig)

            samples_mia_unl = np.concatenate([test_losses_unl, forget_losses_unl])
            labels_mia_unl = np.concatenate([
                np.zeros(len(test_losses_unl)),
                np.ones(len(forget_losses_unl))
            ])
            mia_result_unl = simple_mia(samples_mia_unl, labels_mia_unl)
            
            mia_metrics = {
                "mia_accuracy_original": mia_result_orig["mia_accuracy"],
                "mia_accuracy_unlearned": mia_result_unl["mia_accuracy"],
                "mia_improvement": mia_result_orig["mia_accuracy"] - mia_result_unl["mia_accuracy"]
            }
        except Exception as e:
            print(f"Warning: MIA calculation failed for sklearn: {e}")
            import traceback
            traceback.print_exc()
    
    result.update(mia_metrics)
    #always include MIA keys so downstream comparisons have baselines
    for key in ["mia_accuracy_original", "mia_accuracy_unlearned", "mia_improvement"]:
        if key not in result:
            result[key] = float("nan")

    #activation distance / confidence / JS divergence only make sense for pytorch classifiers
    if is_pytorch_model(original_model) and is_pytorch_model(unlearned_model) and is_classification:
        try:
            activation_metrics = compute_activation_distance(
                model1=original_model,
                model2=unlearned_model,
                data_loader=loader_forget,
                device=device
            )
            result.update(activation_metrics)

            confidence_original = compute_confidence_metrics(
                model=original_model,
                data_loader=loader_forget,
                device=device
            )
            for key, value in confidence_original.items():
                result[f"forget_{key}_original"] = value

            confidence_unlearned = compute_confidence_metrics(
                model=unlearned_model,
                data_loader=loader_forget,
                device=device
            )
            for key, value in confidence_unlearned.items():
                result[f"forget_{key}_unlearned"] = value

            js_metrics = compute_js_divergence(
                model1=original_model,
                model2=unlearned_model,
                data_loader=loader_forget,
                device=device
            )
            result.update(js_metrics)

        except Exception as e:
            print(f"Warning: New metrics calculation failed: {e}")
            import traceback
            traceback.print_exc()

    if is_classification and "activation_cosine_similarity" in result:
        print(f"\n  Unlearning Metrics Summary:")
        print(f"    Forget Accuracy: {result.get('forget_accuracy_original', 'N/A'):.3f} -> "
              f"{result.get('forget_accuracy_unlearned', 'N/A'):.3f} "
              f"(score: {result.get('unlearning_score', 'N/A'):.3f})")
        cos_sim = result.get('activation_cosine_similarity', 1)
        print(f"    Activation Cosine Sim: {cos_sim:.3f} "
              f"({'OK' if cos_sim < 0.6 else 'fail'})")
        conf_unlearned = result.get('forget_confidence_mean_unlearned', 1)
        print(f"    Confidence: {result.get('forget_confidence_mean_original', 'N/A'):.3f} -> "
              f"{conf_unlearned:.3f} "
              f"({'OK' if conf_unlearned < 0.8 else 'fail'})")
        entropy_unlearned = result.get('forget_entropy_mean_unlearned', 0)
        print(f"    Entropy: {result.get('forget_entropy_mean_original', 'N/A'):.3f} -> "
              f"{entropy_unlearned:.3f} "
              f"({'OK' if entropy_unlearned > 0.7 else 'fail'})")
        js_div = result.get('js_divergence_mean', 0)
        print(f"    JS Divergence: {js_div:.3f} "
              f"({'OK' if 0.2 <= js_div <= 0.5 else 'fail'})")
        mia_unlearned = result.get('mia_accuracy_unlearned', 0)
        print(f"    MIA Accuracy: {mia_unlearned:.3f} "
              f"({'OK' if abs(mia_unlearned - 0.5) < 0.1 else 'fail'})")

    return result


def exact_retraining(
    X: np.ndarray,
    y: np.ndarray,
    df,
    input_cols: List[str],
    target_col: str,
    id_column: str,
    forget_ids: Iterable,
    seq_len: int,
    device,
    model_params: Dict = None,
    train_params: Dict = None,
    pretrained_model=None,
    model_type: str = "lstm",
    experiment_type: str = None,
    fl_model_params: Dict = None,
    test_loader: Optional[DataLoader] = None,
    test_df: Optional[pd.DataFrame] = None,
    test_input_cols: Optional[List[str]] = None,
    test_target_col: Optional[str] = None,
    baseline_original_model=None,
) -> Dict:
    model_params = model_params or {}
    train_params = train_params or {}
    fl_model_params = fl_model_params or {}

    validation = _validate_forget_ids(df, id_column, forget_ids)
    print(f"Exact Retraining: Unlearning {validation['n_samples_to_forget']} samples from {validation['n_ids_to_forget']} clients")

    N_train_total = len(df)
    retain_df = drop_rows(df, id_column, forget_ids)
    N_retrain = len(retain_df)
    retrain_fraction = N_retrain / N_train_total if N_train_total > 0 else 0.0

    X_retain, y_retain = extract_features_targets(retain_df, input_cols, target_col)

    if pretrained_model is not None:
        # deepcopy + reinit: reuse FL arch, retrain from scratch, baseline intact
        import copy
        model = copy.deepcopy(pretrained_model)
        if is_pytorch_model(model):
            model.train()
            # Reset ALL trainable layers so retraining starts from scratch (true exact
            # retraining / gold standard). Resetting only Linear layers would leave the
            #convolutional feature extractor warm-started from the pre-unlearning model,
            #which still carries the forgotten client's information.
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, (torch.nn.GroupNorm, torch.nn.BatchNorm2d)):
                    if module.weight is not None:
                        torch.nn.init.ones_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
    else:
        input_size = len(input_cols)
        if experiment_type:
            try:
                import sys
                import os
                fl_module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fl-disagreement-resolution")
                if fl_module_path not in sys.path:
                    sys.path.insert(0, fl_module_path)
                from fl_module import create_model as create_fl_model  # pyright: ignore[reportMissingImports]
                model = create_fl_model(experiment_type=experiment_type, **model_params)
                if device:
                    model = model.to(device)
            except ImportError:
                if model_type == "lstm":
                    model = create_model(model_type, input_size=input_size, device=device, **model_params)
                else:
                    model = create_model(model_type, **model_params)
        else:
            if model_type == "lstm":
                model = create_model(model_type, input_size=input_size, device=device, **model_params)
            else:
                model = create_model(model_type, **model_params)
    
    # training_time_s = only the training loop, for fair comparison across strategies
    training_time_s = 0.0
    if is_pytorch_model(model):
        is_classification = _determine_is_classification(
            experiment_type=experiment_type,
            fl_model_params=fl_model_params,
            model_params=model_params,
            pretrained_model=pretrained_model
        )

        if seq_len == 1 and experiment_type in ["tabular", "adult", "mnist", "cifar10"]:
            from torch.utils.data import TensorDataset
            X_tensor = torch.from_numpy(X_retain).float()

            # reshape flat pixels back to image tensors for the CNN
            if experiment_type == "mnist" and X_tensor.dim() == 2:
                X_tensor = X_tensor.view(-1, 1, 28, 28)
            elif experiment_type == "cifar10" and X_tensor.dim() == 2:
                X_tensor = X_tensor.view(-1, 3, 32, 32)

            y_tensor = torch.from_numpy(y_retain).long() if is_classification else torch.from_numpy(y_retain).float()
            ds_retain = TensorDataset(X_tensor, y_tensor)
        else:
            ds_retain = ArraySequenceDataset(X_retain, y_retain, seq_len=seq_len)
        loader_retain = create_loader(ds_retain, batch_size=train_params.get("batch_size", 64))

        _t0 = time.time()
        model, _ = train_model_universal(
            model,
            train_loader=loader_retain,
            device=device,
            epochs=train_params.get("epochs", 10),
            lr=train_params.get("lr", 1e-3),
            weight_decay=train_params.get("weight_decay", 0.0),
            patience=train_params.get("patience", 5),
            is_classification=is_classification,
        )
        training_time_s = time.time() - _t0
        eval_loader = test_loader if test_loader is not None else loader_retain
        eval_label = "test" if test_loader is not None else "retain"
        metrics_retain = evaluate_model_universal(model, loader=eval_loader, device=device, is_classification=is_classification)
        metrics_utility = _standardize_utility_metrics(metrics_retain, dataset_label=eval_label)
    else:
        #sklearn/XGBoost need sequences built by hand
        X_retain_seq, y_retain_seq = _create_sequences(X_retain, y_retain, seq_len)
        _t0 = time.time()
        model, _ = train_model_universal(
            model,
            X_train=X_retain_seq,
            y_train=y_retain_seq,
        )
        training_time_s = time.time() - _t0
        if test_df is not None and test_input_cols is not None and test_target_col is not None:
            X_test, y_test = extract_features_targets(test_df, test_input_cols, test_target_col)
            eval_label = "test"
            metrics_retain = evaluate_model_universal(model, X=X_test, y=y_test)
        else:
            eval_label = "retain"
            metrics_retain = evaluate_model_universal(model, X=X_retain_seq, y=y_retain_seq)
        metrics_utility = _standardize_utility_metrics(metrics_retain, dataset_label=eval_label)
    
    #baseline_original_model: pre-exclusion FL model, the unlearning reference
    metrics_forget = {}
    if len(validation['found_ids']) > 0:
        original_model_for_comparison = baseline_original_model or pretrained_model
        if original_model_for_comparison is not None:
            metrics_forget = _evaluate_forget_set(
                original_model=original_model_for_comparison,
                unlearned_model=model,
                df=df,
                input_cols=input_cols,
                target_col=target_col,
                id_column=id_column,
                forget_ids=forget_ids,
                seq_len=seq_len,
                device=device,
                train_params=train_params,
                experiment_type=experiment_type,
                is_classification=is_classification,
                test_loader=test_loader,
                test_df=test_df,
                test_input_cols=test_input_cols,
                test_target_col=test_target_col
            )
    
    behavioral_distance_metrics = {}
    if test_loader is not None and pretrained_model is not None:
        try:
            is_classification = _determine_is_classification(
                experiment_type=experiment_type,
                fl_model_params=fl_model_params,
                model_params=model_params,
                pretrained_model=pretrained_model
            )
            behavioral_distance_metrics = calculate_behavioral_distance(
                model_unlearned=model,
                model_baseline=pretrained_model,
                loader=test_loader,
                device=device,
                is_classification=is_classification
            )
        except Exception as e:
            print(f"Warning: Could not calculate behavioral distance: {e}")

    # training_time_s = time spent only in train step, not data prep/eval
    efficiency_metrics = {
        "retrain_fraction": retrain_fraction,
        "N_train_total": N_train_total,
        "N_retrain": N_retrain,
        "training_time_s": training_time_s
    }

    return {
        "model": model,
        "metrics_utility": metrics_utility,
        "metrics_retain": metrics_utility,
        "metrics_forget": metrics_forget,
        "efficiency_metrics": efficiency_metrics,
        "behavioral_distance": behavioral_distance_metrics
    }


def aggregate_sisa_models(models: List[Dict], model_type: str = None) -> Union[torch.nn.Module, object]:
    """Aggregate SISA submodels into a prediction-averaging ensemble."""
    if not models:
        raise ValueError("No models to aggregate")

    # use the final slice per shard (highest slice index, ignore initial checkpoints)
    shard_latest = {}
    for m in models:
        shard, sl = m["meta"]
        if sl is None or sl < 0:
            continue
        if shard not in shard_latest or sl > shard_latest[shard]["slice"]:
            shard_latest[shard] = {"slice": sl, "model": m["model"]}
    
    final_models = [v["model"] for _, v in sorted(shard_latest.items(), key=lambda kv: kv[0])]
    if not final_models:
        raise ValueError("No shard models available for aggregation")
    
    first_model = final_models[0]
    is_pytorch = is_pytorch_model(first_model)

    if model_type is None:
        if is_pytorch:
            model_type = "pytorch"
        elif is_sklearn_model(first_model):
            from xgboost import XGBRegressor
            if isinstance(first_model, XGBRegressor):
                model_type = "xgboost"
            else:
                model_type = "random_forest"
    
    if is_pytorch or model_type in ["pytorch", "lstm"]:
        import torch.nn as nn

        class SISAEnsemble(nn.Module):
            def __init__(self, models):
                super().__init__()
                self.models = nn.ModuleList(models)

            def forward(self, x):
                # average softmax probs per shard model (Bourtoule et al. 2021)
                outputs = [torch.softmax(m(x), dim=-1) for m in self.models]
                stacked = torch.stack(outputs, dim=0)  #[num_models, batch, num_classes]
                return torch.mean(stacked, dim=0)

        return SISAEnsemble(final_models)
    else:
        class PredictionAveragingEnsemble:
            def __init__(self, models):
                self.models = models
            
            def predict(self, X):
                predictions = [model.predict(X) for model in self.models]
                return np.mean(predictions, axis=0)
            
            def predict_proba(self, X):
                if hasattr(self.models[0], 'predict_proba'):
                    probas = [model.predict_proba(X) for model in self.models]
                    return np.mean(probas, axis=0)
                else:
                    raise AttributeError("predict_proba not available")
        
        return PredictionAveragingEnsemble(final_models)


def sisa_unlearning(
    X: np.ndarray,
    y: np.ndarray,
    df,
    input_cols: List[str],
    target_col: str,
    id_column: str,
    forget_ids: Iterable,
    seq_len: int,
    device,
    num_shards: int = 2,
    num_slices: int = 2,
    model_params: Dict = None,
    train_params: Dict = None,
    random_state: int = 42,
    pretrained_models: List = None,
    pretrained_model=None,
    baseline_model=None,
    baseline_original_model=None,
    model_type: str = "lstm",
    experiment_type: str = None,
    fl_model_params: Dict = None,
    test_loader: Optional[DataLoader] = None,
    test_df: Optional[pd.DataFrame] = None,
    test_input_cols: Optional[List[str]] = None,
    test_target_col: Optional[str] = None,
) -> Dict:
    import copy
    from collections import defaultdict

    model_params = model_params or {}
    train_params = train_params or {}
    fl_model_params = fl_model_params or {}
    random_state = train_params.get("random_state", random_state)
    client_aware = train_params.get("client_aware_sharding", False)

    validation = _validate_forget_ids(df, id_column, forget_ids)

    print(f"\n{'='*60}")
    print(f"SISA Unlearning Started")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"Forget IDs: {sorted(list(forget_ids))}")
    print(f"Shards: {num_shards}, Slices per shard: {num_slices}")
    total_slices = num_shards * num_slices
    print(f"Total slices: {total_slices}")
    print(f"Pretrained models available: {pretrained_models is not None}")
    if pretrained_models is not None:
        print(f"Number of pretrained models: {len(pretrained_models)}")

    N_train_total = len(df)
    slices = split_into_slices(
        X=X,
        y=y,
        df=df,
        id_column=id_column,
        num_shards=num_shards,
        num_slices=num_slices,
        random_state=random_state,
        client_aware=client_aware,
    )
    input_size = len(input_cols)
    forget_set = set(forget_ids)

    #group slices per shard, sorted by slice index
    shard_slices: Dict[int, List[Dict]] = defaultdict(list)
    for s in slices:
        shard_slices[s["shard"]].append(s)
    for shard_idx in shard_slices:
        shard_slices[shard_idx] = sorted(shard_slices[shard_idx], key=lambda s: s["slice"])

    # every slice in current data + init (-1) per shard must exist for reuse to be valid
    required_slice_keys = (
        set((s["shard"], s["slice"]) for s in slices)
        | {(shard_idx, -1) for shard_idx in shard_slices.keys()}
    )
    if pretrained_models:
        have_keys = set()
        for m in pretrained_models:
            if m is None:
                continue
            meta = getattr(m, "meta", m.get("meta")) if isinstance(m, dict) else getattr(m, "meta", None)
            if meta is not None and meta != (None, None):
                have_keys.add(meta)
        if not required_slice_keys.issubset(have_keys):
            missing = required_slice_keys - have_keys
            print(
                f"SISA: pretrained slice layout does not match current data (missing {missing}), "
                "training baseline from scratch for this track"
            )
            pretrained_models = None

    def _unwrap_model(maybe_model):
        if isinstance(maybe_model, dict) and "model" in maybe_model:
            return maybe_model["model"]
        return maybe_model

    def _init_model():
        if experiment_type:
            try:
                import sys
                import os
                fl_module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fl-disagreement-resolution")
                if fl_module_path not in sys.path:
                    sys.path.insert(0, fl_module_path)
                from fl_module import create_model as create_fl_model  # pyright: ignore[reportMissingImports]
                merged_params = {**fl_model_params, **model_params}
                model = create_fl_model(experiment_type=experiment_type, **merged_params)
                if device:
                    model = model.to(device)
                return model
            except ImportError:
                pass

        if model_type == "lstm":
            return create_model(model_type, input_size=input_size, device=device, **model_params)
        else:
            params = dict(model_params)
            if "input_size" not in params and input_size:
                params["input_size"] = input_size
            return create_model(model_type, **params)

    def _make_loader(X_slice: np.ndarray, y_slice: np.ndarray, is_classification: bool):
        if seq_len == 1 and experiment_type in ["tabular", "adult", "mnist", "cifar10"]:
            from torch.utils.data import TensorDataset
            X_tensor = torch.from_numpy(X_slice).float()
            if experiment_type == "mnist" and X_tensor.dim() == 2:
                X_tensor = X_tensor.view(-1, 1, 28, 28)
            elif experiment_type == "cifar10" and X_tensor.dim() == 2:
                X_tensor = X_tensor.view(-1, 3, 32, 32)
            y_tensor = torch.from_numpy(y_slice).long() if is_classification else torch.from_numpy(y_slice).float()
            ds = TensorDataset(X_tensor, y_tensor)
        else:
            ds = ArraySequenceDataset(X_slice, y_slice, seq_len=seq_len)
        return create_loader(ds, batch_size=train_params.get("batch_size", 64))

    # baseline slice models still include the forget data
    baseline_models = []
    if pretrained_models:
        baseline_models = [
            {"model": _unwrap_model(m), "meta": getattr(m, "meta", m.get("meta")) if isinstance(m, dict) else (None, None), "ids": set(m.get("ids", [])) if isinstance(m, dict) else set()}
            if m is not None else None
            for m in pretrained_models
        ]
        baseline_models = [m for m in baseline_models if m is not None]
    else:
        print("No SISA checkpoints found - training baseline SISA shard/slice models")
        for shard_idx in sorted(shard_slices.keys()):
            shard_data = shard_slices[shard_idx]
            model = _init_model()
            #initial checkpoint per shard is slice -1
            baseline_models.append({"model": copy.deepcopy(model), "meta": (shard_idx, -1), "ids": set(), "trained_from_scratch": True})
            is_classification = _determine_is_classification(
                experiment_type=experiment_type,
                fl_model_params=fl_model_params,
                model_params=model_params,
                pretrained_model=pretrained_model
            )
            for s in shard_data:
                X_slice = s["X"]
                y_slice = s["y"]
                loader = _make_loader(X_slice, y_slice, is_classification)
                model, _ = train_model_universal(
                    model,
                    train_loader=loader,
                    device=device,
                    epochs=train_params.get("epochs", 10),
                    lr=train_params.get("lr", 1e-3),
                    weight_decay=train_params.get("weight_decay", 0.0),
                    patience=train_params.get("patience", 5),
                    is_classification=is_classification,
                )
                baseline_models.append({
                    "model": copy.deepcopy(model),
                    "meta": (s["shard"], s["slice"]),
                    "ids": set(s["ids"]),
                    "trained_from_scratch": True
                })

    baseline_map: Dict[Tuple[int, int], Dict] = {}
    for m in baseline_models:
        baseline_map[m["meta"]] = m

    #which slices contain forget ids
    affected_shards: Dict[int, List[int]] = defaultdict(list)
    for s in slices:
        if set(s["ids"]).intersection(forget_set):
            affected_shards[s["shard"]].append(s["slice"])

    # nothing to forget: just aggregate the baseline
    if not forget_set or all(len(v) == 0 for v in affected_shards.values()):
        ensemble_model = aggregate_sisa_models(baseline_models, model_type=model_type)
        selected_models = baseline_models
        affected = []
        N_retrain = 0
        training_time_s = 0.0
    else:
        selected_models = []
        affected = []
        N_retrain = 0
        training_time_s = 0.0

        for shard_idx in sorted(shard_slices.keys()):
            shard_data = shard_slices[shard_idx]
            shard_aff = sorted(set(affected_shards.get(shard_idx, [])))
            is_classification = _determine_is_classification(
                experiment_type=experiment_type,
                fl_model_params=fl_model_params,
                model_params=model_params,
                pretrained_model=pretrained_model
            )

            if not shard_aff:
                # unaffected shard: reuse its slice models as-is
                selected_models.extend([m for m in baseline_models if m["meta"][0] == shard_idx])
                continue

            start_slice = min(shard_aff)
            affected.extend([idx for idx, s in enumerate(slices) if s["shard"] == shard_idx and s["slice"] in shard_aff])

            # retrain from the slice before the first affected one (or init -1)
            start_meta = (shard_idx, start_slice - 1)
            start_model = baseline_map.get(start_meta)
            if start_model is None:
                start_model = baseline_map.get((shard_idx, -1))
            if start_model is None:
                start_model = {"model": _init_model(), "meta": (shard_idx, -1), "ids": set()}

            current_model = copy.deepcopy(start_model["model"])
            if is_pytorch_model(current_model):
                current_model.train()

            #keep the pre-retrain checkpoint for traceability
            selected_models.append({
                "model": copy.deepcopy(current_model),
                "meta": start_model["meta"],
                "ids": set(),
                "trained_from_scratch": True
            })

            for s in shard_data:
                if s["slice"] < start_slice:
                    #unaffected prefix: reuse baseline slice
                    selected_models.append(baseline_map[(shard_idx, s["slice"])])
                    continue

                ids_in_slice = list(s["ids"])
                retain_mask = [i for i, _id in enumerate(ids_in_slice) if _id not in forget_set]
                if not retain_mask:
                    # whole slice forgotten; keep a placeholder so the saved layout still matches required_slice_keys
                    print(f"SISA slice shard {shard_idx} slice {s['slice']}: all data forgotten, reusing previous slice model for layout.")
                    selected_models.append({
                        "model": copy.deepcopy(current_model),
                        "meta": (s["shard"], s["slice"]),
                        "ids": set(),
                        "trained_from_scratch": False,
                    })
                    continue

                X_slice = s["X"][retain_mask]
                y_slice = s["y"][retain_mask]
                ids_slice = [ids_in_slice[i] for i in retain_mask]
                N_retrain += len(ids_slice)

                loader = _make_loader(X_slice, y_slice, is_classification)
                _t0 = time.time()
                current_model, _ = train_model_universal(
                    current_model,
                    train_loader=loader,
                    device=device,
                    epochs=train_params.get("epochs", 10),
                    lr=train_params.get("lr", 1e-3),
                    weight_decay=train_params.get("weight_decay", 0.0),
                    patience=train_params.get("patience", 5),
                    is_classification=is_classification,
                )
                training_time_s += time.time() - _t0

                selected_models.append({
                    "model": copy.deepcopy(current_model),
                    "meta": (s["shard"], s["slice"]),
                    "ids": set(ids_slice),
                    "trained_from_scratch": True
                })

        retrain_fraction = N_retrain / N_train_total if N_train_total > 0 else 0.0
        print(f"\nSISA retraining: retrain_fraction={retrain_fraction:.4f}, N_retrain={N_retrain}, N_total={N_train_total}")
        ensemble_model = aggregate_sisa_models(selected_models, model_type=model_type)

    is_classification = _determine_is_classification(
        experiment_type=experiment_type,
        fl_model_params=fl_model_params,
        model_params=model_params,
        pretrained_model=pretrained_model
    )

    metrics_retain = {}
    metrics_utility = {}
    retain_df = drop_rows(df, id_column, forget_ids)
    if len(retain_df) > 0:
        X_retain, y_retain = extract_features_targets(retain_df, input_cols, target_col)
        
        if is_pytorch_model(ensemble_model):
            if seq_len == 1 and experiment_type in ["tabular", "adult", "mnist", "cifar10"]:
                from torch.utils.data import TensorDataset
                X_retain_tensor = torch.from_numpy(X_retain).float()
                if experiment_type == "mnist" and X_retain_tensor.dim() == 2:
                    X_retain_tensor = X_retain_tensor.view(-1, 1, 28, 28)
                elif experiment_type == "cifar10" and X_retain_tensor.dim() == 2:
                    X_retain_tensor = X_retain_tensor.view(-1, 3, 32, 32)
                y_retain_tensor = torch.from_numpy(y_retain).long() if is_classification else torch.from_numpy(y_retain).float()
                ds_retain = TensorDataset(X_retain_tensor, y_retain_tensor)
            else:
                ds_retain = ArraySequenceDataset(X_retain, y_retain, seq_len=seq_len)
            loader_retain = create_loader(ds_retain, batch_size=train_params.get("batch_size", 64))
            eval_loader = test_loader if test_loader is not None else loader_retain
            eval_label = "test" if test_loader is not None else "retain"
            metrics_retain = evaluate_model_universal(ensemble_model, loader=eval_loader, device=device, is_classification=is_classification)
            metrics_utility = _standardize_utility_metrics(metrics_retain, dataset_label=eval_label)
        else:
            # sklearn models
            X_retain_seq, y_retain_seq = _create_sequences(X_retain, y_retain, seq_len)
            if test_df is not None and test_input_cols is not None and test_target_col is not None:
                X_test, y_test = extract_features_targets(test_df, test_input_cols, test_target_col)
                eval_label = "test"
                metrics_retain = evaluate_model_universal(ensemble_model, X=X_test, y=y_test)
            else:
                eval_label = "retain"
                metrics_retain = evaluate_model_universal(ensemble_model, X=X_retain_seq, y=y_retain_seq)
            metrics_utility = _standardize_utility_metrics(metrics_retain, dataset_label=eval_label)
    
    # forget-set eval: baseline_original_model if present, else pretrained SISA
    metrics_forget = {}
    if baseline_original_model is not None:
        original_model = baseline_original_model
    elif baseline_models:
        try:
            original_model = aggregate_sisa_models(baseline_models, model_type=model_type)
        except Exception:
            original_model = None
            print("Warning: Could not build baseline SISA ensemble for evaluation")
    else:
        original_model = None
        print("Warning: No baseline model available for SISA evaluation")
    
    if original_model is not None and len(validation['found_ids']) > 0:
        metrics_forget = _evaluate_forget_set(
            original_model=original_model,
            unlearned_model=ensemble_model,
            df=df,
            input_cols=input_cols,
            target_col=target_col,
            id_column=id_column,
            forget_ids=forget_ids,
            seq_len=seq_len,
            device=device,
            train_params=train_params,
            experiment_type=experiment_type,
            is_classification=is_classification,
            test_loader=test_loader,
            test_df=test_df,
            test_input_cols=test_input_cols,
            test_target_col=test_target_col
        )
    
    #Calculate behavioral distance if test data is available
    behavioral_distance_metrics = {}
    if test_loader is not None and original_model is not None:
        try:
            is_classification = _determine_is_classification(
                experiment_type=experiment_type,
                fl_model_params=fl_model_params,
                model_params=model_params,
                pretrained_model=pretrained_model
            )
            behavioral_distance_metrics = calculate_behavioral_distance(
                model_unlearned=ensemble_model,
                model_baseline=original_model,  #Baseline is the pre-unlearning model
                loader=test_loader,
                device=device,
                is_classification=is_classification
            )
        except Exception as e:
            print(f"Warning: Could not calculate behavioral distance for SISA: {e}")
    
    # Efficiency metrics (training_time_s = only slice training loops, not load/save/aggregate)
    efficiency_metrics = {
        "retrain_fraction": (N_retrain / N_train_total) if N_train_total > 0 else 0.0,
        "N_train_total": N_train_total,
        "N_retrain": N_retrain,
        "num_affected_slices": len(affected),
        "total_slices": len(slices),
        "training_time_s": training_time_s
    }

    return {
        "models": selected_models if selected_models else baseline_models,
        "affected_slices": affected,
        "ensemble": ensemble_model,
        "metrics_utility": metrics_utility,
        "metrics_retain": metrics_utility,
        "metrics_forget": metrics_forget,
        "efficiency_metrics": efficiency_metrics,
        "behavioral_distance": behavioral_distance_metrics
    }


def knowledge_distillation(
    X: np.ndarray,
    y: np.ndarray,
    df,
    input_cols: List[str],
    target_col: str,
    id_column: str,
    forget_ids: Iterable,
    seq_len: int,
    device,
    teacher_params: Dict = None,
    student_params: Dict = None,
    train_params: Dict = None,
    pretrained_teacher=None,
    baseline_model=None,
    baseline_original_model=None,
    model_type: str = "lstm",
    experiment_type: str = None,
    fl_model_params: Dict = None,
    test_loader: Optional[DataLoader] = None,
    test_df: Optional[pd.DataFrame] = None,
    test_input_cols: Optional[List[str]] = None,
    test_target_col: Optional[str] = None,
    use_federaser: bool = False,
    results_dir: Optional[str] = None,
    num_clients: Optional[int] = None,
    current_round: Optional[int] = None,
) -> Dict:
    # Ensure TensorDataset is always in scope for all branches
    from torch.utils.data import TensorDataset
    teacher_params = teacher_params or {}
    student_params = student_params or {}
    train_params = train_params or {}
    mf_mode = train_params.get("mf_mode", False)

    # Validate forget_ids
    validation = _validate_forget_ids(df, id_column, forget_ids)
    print(f"Knowledge Distillation: Unlearning {validation['n_samples_to_forget']} samples from {validation['n_ids_to_forget']} clients")

    #Retrain fraction berekenen
    N_train_total = len(df)
    
    input_size = len(input_cols)
    
    #Teacher
    teacher = None
    if baseline_model is not None:
        import copy
        teacher = copy.deepcopy(baseline_model)
        if is_pytorch_model(teacher):
            teacher.eval()
    elif not mf_mode:
        retain_df_teacher = exclude_ids(df, id_column, forget_ids)
        X_retain_teacher, y_retain_teacher = extract_features_targets(retain_df_teacher, input_cols, target_col)

        if model_type == "lstm":
            teacher = create_model(model_type, input_size=input_size, device=device, **teacher_params)

            if seq_len == 1 and experiment_type in ["tabular", "adult", "mnist", "cifar10"]:
                X_retain_teacher_tensor = torch.from_numpy(X_retain_teacher).float()
                if experiment_type == "mnist" and X_retain_teacher_tensor.dim() == 2:
                    X_retain_teacher_tensor = X_retain_teacher_tensor.view(-1, 1, 28, 28)
                elif experiment_type == "cifar10" and X_retain_teacher_tensor.dim() == 2:
                    X_retain_teacher_tensor = X_retain_teacher_tensor.view(-1, 3, 32, 32)

                is_classification = _determine_is_classification(
                    experiment_type=experiment_type,
                    fl_model_params=fl_model_params,
                    model_params=teacher_params,
                    pretrained_model=pretrained_teacher
                )
                y_retain_teacher_tensor = torch.from_numpy(y_retain_teacher).long() if is_classification else torch.from_numpy(y_retain_teacher).float()
                ds_retain_teacher = TensorDataset(X_retain_teacher_tensor, y_retain_teacher_tensor)
            else:
                ds_retain_teacher = ArraySequenceDataset(X_retain_teacher, y_retain_teacher, seq_len=seq_len)

            loader_retain_teacher = create_loader(ds_retain_teacher, batch_size=train_params.get("batch_size", 64))

            teacher, _ = train_model_universal(
                teacher,
                train_loader=loader_retain_teacher,
                device=device,
                epochs=train_params.get("teacher_epochs", 10),
                lr=train_params.get("lr", 1e-3),
                weight_decay=train_params.get("weight_decay", 0.0),
                patience=train_params.get("patience", 5),
                is_classification=is_classification,
            )
        else:
            X_retain_teacher_seq, y_retain_teacher_seq = _create_sequences(X_retain_teacher, y_retain_teacher, seq_len)
            teacher_params_with_input = dict(teacher_params)
            if model_type == "mlp" and "input_size" not in teacher_params_with_input:
                teacher_params_with_input["input_size"] = input_size
            teacher = create_model(model_type, **teacher_params_with_input)
            teacher, _ = train_model_universal(
                teacher,
                X_train=X_retain_teacher_seq,
                y_train=y_retain_teacher_seq,
            )
    else:
        teacher = pretrained_teacher
        if is_pytorch_model(teacher):
            teacher.eval()

    # When FedEraser is on, the student starts at M'F = MF - (1/N) * Σ(ΔM_forget)
    # instead of from scratch. The regular KD loop below then repairs M'F.
    student_init_params = None
    federaser_metadata = {}

    if use_federaser and results_dir is not None and num_clients is not None:
        print("FedEraser: computing damaged student M'F before distillation")

        try:
            from .federaser_utils import compute_client_total_contribution, scale_delta, subtract_delta_from_model, sum_deltas

            # MF is the FL model from before unlearning. Prefer baseline_original_model;
            #fall back to baseline_model if the caller didn't pass it.
            if baseline_original_model is not None:
                import copy
                mf_model = copy.deepcopy(baseline_original_model)
                print("  using baseline_original_model as MF")
            elif baseline_model is not None:
                import copy
                mf_model = copy.deepcopy(baseline_model)
                print("  using baseline_model as MF (fallback)")
            else:
                print("Warning: No MF model available, cannot use FedEraser")
                use_federaser = False
                mf_model = None

            if use_federaser and mf_model is not None:
                #Sum each forget client's total ΔM contribution across all FL rounds.
                all_deltas = []
                for forget_id in forget_ids:
                    print(f"  computing ΔM for forget client {forget_id}")
                    delta, meta = compute_client_total_contribution(
                        results_dir=results_dir,
                        forget_client_id=forget_id,
                        device=device,
                        experiment_type=experiment_type
                    )

                    if delta:
                        all_deltas.append(delta)
                        federaser_metadata[f"client_{forget_id}"] = meta
                    else:
                        print(f"Warning: No delta computed for client {forget_id}")

                if all_deltas:
                    # sum, scale by 1/N, subtract from MF -> M'F
                    total_delta = sum_deltas(all_deltas)
                    scaled_delta = scale_delta(total_delta, 1.0 / num_clients)
                    mf_damaged = subtract_delta_from_model(mf_model, scaled_delta)

                    # The student gets initialised from these parameters further down.
                    student_init_params = [p.data.clone() for p in mf_damaged.parameters()]

                    print(f"  M'F ready: forget={list(forget_ids)}, N={num_clients}, "
                          f"deltas={len(all_deltas)}")
                else:
                    print("Warning: No deltas computed, falling back to standard distillation")
                    use_federaser = False

        except Exception as e:
            print(f"Error in FedEraser initialization: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to standard distillation")
            use_federaser = False

    # Retain set ophalen (alleen voor metrics); KD-data kan extern zijn
    retain_df = exclude_ids(df, id_column, forget_ids)
    N_retrain = len(retain_df)
    retrain_fraction = N_retrain / N_train_total if N_train_total > 0 else 0.0
    X_retain, y_retain = extract_features_targets(retain_df, input_cols, target_col)

    is_classification = _determine_is_classification(
        experiment_type=experiment_type,
        fl_model_params=fl_model_params,
        model_params=student_params,
        pretrained_model=pretrained_teacher
    )
    metrics_utility: Dict[str, float] = {}
    eval_label = "test" if test_loader is not None else "retain"

    #KD-data bron kiezen
    if mf_mode:
        #Externe unlabeled data (random noise)
        unlabeled_samples = train_params.get("mf_unlabeled_samples", 2000)
        if seq_len == 1 and experiment_type == "mnist":
            ext_x = torch.randn(unlabeled_samples, 1, 28, 28)
        elif seq_len == 1 and experiment_type == "cifar10":
            ext_x = torch.randn(unlabeled_samples, 3, 32, 32)
        elif seq_len == 1:
            ext_x = torch.randn(unlabeled_samples, len(input_cols))
        else:
            ext_x = torch.randn(unlabeled_samples, seq_len, len(input_cols))
        ext_ds = TensorDataset(ext_x, torch.zeros(unlabeled_samples))
        kd_loader_source = DataLoader(ext_ds, batch_size=train_params.get("batch_size", 64), shuffle=False)
        N_retrain = unlabeled_samples
        retrain_fraction = N_retrain / N_train_total if N_train_total > 0 else 0.0
    else:
        if is_pytorch_model(teacher):
            if seq_len == 1 and experiment_type in ["tabular", "adult", "mnist", "cifar10"]:
                X_retain_tensor = torch.from_numpy(X_retain).float()
                if experiment_type == "mnist" and X_retain_tensor.dim() == 2:
                    X_retain_tensor = X_retain_tensor.view(-1, 1, 28, 28)
                elif experiment_type == "cifar10" and X_retain_tensor.dim() == 2:
                    X_retain_tensor = X_retain_tensor.view(-1, 3, 32, 32)
                y_retain_tensor = torch.from_numpy(y_retain).long() if is_classification else torch.from_numpy(y_retain).float()
                ds_retain = TensorDataset(X_retain_tensor, y_retain_tensor)
            else:
                ds_retain = ArraySequenceDataset(X_retain, y_retain, seq_len=seq_len)
            kd_loader_source = create_loader(ds_retain, batch_size=train_params.get("batch_size", 64), shuffle=False)
        else:
            X_retain_seq, y_retain_seq = _create_sequences(X_retain, y_retain, seq_len)
            kd_loader_source = create_loader(ArraySequenceDataset(X_retain_seq, y_retain_seq, seq_len=1), batch_size=train_params.get("batch_size", 64), shuffle=False)

    # Generate soft targets from teacher
    if is_pytorch_model(teacher):
        teacher.eval()
        y_soft_list = []
        x_list = []
        with torch.no_grad():
            for batch in kd_loader_source:
                xb = batch[0].to(device)
                preds = teacher(xb)
                # Store raw logits; train_with_soft_labels applies softmax(logits/T)
                # per Hinton et al. (2015). Do NOT pre-softmax here.
                y_soft_list.append(preds.detach().cpu())
                x_list.append(xb.cpu())

        y_soft = torch.cat(y_soft_list, dim=0) if y_soft_list else torch.empty(0)
        X_soft = torch.cat(x_list, dim=0) if x_list else torch.empty(0)
        soft_ds = TensorDataset(X_soft, y_soft)
        soft_loader = DataLoader(soft_ds, batch_size=train_params.get("batch_size", 64), shuffle=True)
        
        #Gebruik FL model factory als experiment_type beschikbaar is
        if experiment_type:
            try:
                import sys
                import os
                fl_module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fl-disagreement-resolution")
                if fl_module_path not in sys.path:
                    sys.path.insert(0, fl_module_path)
                from fl_module import create_model as create_fl_model  # pyright: ignore[reportMissingImports]
                # Merge fl_model_params met student_params
                fl_model_params = fl_model_params or {}
                merged_params = {**fl_model_params, **student_params}
                student = create_fl_model(experiment_type=experiment_type, **merged_params)
                if device:
                    student = student.to(device)
            except ImportError:
                # Fallback naar unlearning framework models
                student = create_model(model_type, input_size=input_size, device=device, **student_params)
        else:
            student = create_model(model_type, input_size=input_size, device=device, **student_params)

        # FEDERASER: Initialize with M'F if computed
        if use_federaser and student_init_params is not None:
            print("Initializing student with M'F (damaged model)")
            if hasattr(student, 'set_parameters'):
                student.set_parameters(student_init_params)
            else:
                #Fallback: direct parameter assignment
                for param, init_param in zip(student.parameters(), student_init_params):
                    param.data = init_param.clone()
            print("Student initialized with M'F")
        else:
            print("Student initialized from scratch (standard distillation)")

        _t0 = time.time()
        student = train_with_soft_labels(
            student,
            train_loader=soft_loader,
            device=device,
            epochs=train_params.get("student_epochs", 10),
            lr=train_params.get("lr", 1e-3),
            weight_decay=train_params.get("weight_decay", 0.0),
            alpha=train_params.get("alpha", 1.0),
            is_classification=is_classification,
            temperature=train_params.get("temperature", 3.0),
        )
        training_time_s = time.time() - _t0
        #Evaluatie: gebruik test_loader als die bestaat, anders de KD bron (retain/noise)
        eval_loader = test_loader if test_loader is not None else kd_loader_source
        eval_label = "test" if test_loader is not None else "kd"
        metrics_retain = evaluate_model_universal(student, loader=eval_loader, device=device, is_classification=is_classification)
        metrics_utility = _standardize_utility_metrics(metrics_retain, dataset_label=eval_label)
    else:
        # For tree models: use teacher predictions as soft targets
        X_retain_seq, y_retain_seq = _create_sequences(X_retain, y_retain, seq_len)

        is_classification = _determine_is_classification(
            experiment_type=experiment_type,
            fl_model_params=fl_model_params,
            model_params=teacher_params,
            pretrained_model=pretrained_teacher
        )

        if is_classification and hasattr(teacher, 'predict_proba'):
            # Use probability distributions for better soft label quality
            y_soft = teacher.predict_proba(X_retain_seq)
        else:
            y_soft = teacher.predict(X_retain_seq)

        # Train student on soft targets
        student = create_model(model_type, **student_params)
        _t0 = time.time()
        student.fit(X_retain_seq, y_soft)  #Train on teacher predictions/probabilities
        training_time_s = time.time() - _t0
        if test_df is not None and test_input_cols is not None and test_target_col is not None:
            X_test, y_test = extract_features_targets(test_df, test_input_cols, test_target_col)
            eval_label = "test"
            metrics_retain = evaluate_model_universal(student, X=X_test, y=y_test)
        else:
            eval_label = "kd"
            metrics_retain = evaluate_model_universal(student, X=X_retain_seq, y=y_retain_seq)
        metrics_utility = _standardize_utility_metrics(metrics_retain, dataset_label=eval_label)
    
    #forget-set eval: only with a baseline or teacher, and
    # forget_ids is non-empty. Preferred is baseline_original_model (the FL
    # model from before exclusion); the teacher (trained on the full set)
    # serves as fallback.
    metrics_forget = {}
    original_model_for_eval = baseline_original_model or teacher or pretrained_teacher

    if original_model_for_eval is None:
        print("Warning: No baseline model available for distillation evaluation")

    if original_model_for_eval is not None and len(validation['found_ids']) > 0:
        is_classification = _determine_is_classification(
            experiment_type=experiment_type,
            fl_model_params=fl_model_params,
            model_params=student_params,
            pretrained_model=original_model_for_eval
        )
        metrics_forget = _evaluate_forget_set(
            original_model=original_model_for_eval,
            unlearned_model=student,
            df=df,
            input_cols=input_cols,
            target_col=target_col,
            id_column=id_column,
            forget_ids=forget_ids,
            seq_len=seq_len,
            device=device,
            train_params=train_params,
            experiment_type=experiment_type,
            is_classification=is_classification,
            test_loader=test_loader,
            test_df=test_df,
            test_input_cols=test_input_cols,
            test_target_col=test_target_col
        )
    
    #Calculate behavioral distance if data is available
    #Use baseline_model for behavioral distance if available
    behavioral_distance_metrics = {}
    baseline_for_distance = baseline_model if baseline_model is not None else pretrained_teacher
    if test_loader is not None and baseline_for_distance is not None:
        try:
            is_classification = _determine_is_classification(
                experiment_type=experiment_type,
                fl_model_params=fl_model_params,
                model_params=student_params,
                pretrained_model=baseline_for_distance
            )
            behavioral_distance_metrics = calculate_behavioral_distance(
                model_unlearned=student,
                model_baseline=baseline_for_distance,  # Use Exact Retraining as baseline
                loader=test_loader,
                device=device,
                is_classification=is_classification
            )
        except Exception as e:
            print(f"Warning: Could not calculate behavioral distance for distillation: {e}")
    
    # Efficiency metrics (training_time_s = only student training, not teacher/soft-label prep)
    efficiency_metrics = {
        "retrain_fraction": retrain_fraction,
        "N_train_total": N_train_total,
        "N_retrain": N_retrain,
        "training_time_s": training_time_s,
        "note": f"Utility eval on {eval_label}; KD source={'noise' if mf_mode else 'retain'}; teacher delivers soft labels"
    }

    return {
        "teacher": teacher,
        "model": student,  # Use "model" for consistency with other strategies
        "metrics_utility": metrics_utility,
        "metrics_retain": metrics_utility,
        "metrics_forget": metrics_forget,
        "efficiency_metrics": efficiency_metrics,
        "behavioral_distance": behavioral_distance_metrics,
        "federaser_enabled": use_federaser,
        "federaser_metadata": federaser_metadata if use_federaser else {}
    }
