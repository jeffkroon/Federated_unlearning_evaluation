from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .unlearning import exact_retraining, knowledge_distillation, sisa_unlearning


def _device_from_arg(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run_exact_retraining(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str,
    id_column: str,
    forget_ids: Iterable,
    device: Optional[str] = None,
    seq_len: int = 24,
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
    device_t = _device_from_arg(device)
    result = exact_retraining(
        X=X,
        y=y,
        df=df,
        input_cols=input_cols,
        target_col=target_col,
        id_column=id_column,
        forget_ids=forget_ids,
        seq_len=seq_len,
        device=device_t,
        model_params=model_params or {},
        train_params=train_params or {},
        pretrained_model=pretrained_model,
        experiment_type=experiment_type,
        model_type=model_type,
        fl_model_params=fl_model_params or {},
        test_loader=test_loader,
        test_df=test_df,
        test_input_cols=test_input_cols,
        test_target_col=test_target_col,
        baseline_original_model=baseline_original_model,
    )
    return result


def run_sisa_unlearning(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str,
    id_column: str,
    forget_ids: Iterable,
    device: Optional[str] = None,
    seq_len: int = 24,
    num_shards: int = 2,
    num_slices: int = 2,
    model_params: Dict = None,
    train_params: Dict = None,
    random_state: int = 42,
    pretrained_models: List = None,
    pretrained_model=None,
    baseline_model=None,
    model_type: str = "lstm",
    experiment_type: str = None,
    fl_model_params: Dict = None,
    test_loader: Optional[DataLoader] = None,
    test_df: Optional[pd.DataFrame] = None,
    test_input_cols: Optional[List[str]] = None,
    test_target_col: Optional[str] = None,
    baseline_original_model=None,
) -> Dict:
    device_t = _device_from_arg(device)
    #prefer the exact-retraining baseline for evaluation, else the pretrained model
    baseline_for_eval = baseline_model if baseline_model is not None else pretrained_model
    result = sisa_unlearning(
        X=X,
        y=y,
        df=df,
        input_cols=input_cols,
        target_col=target_col,
        id_column=id_column,
        forget_ids=forget_ids,
        seq_len=seq_len,
        device=device_t,
        num_shards=num_shards,
        num_slices=num_slices,
        model_params=model_params or {},
        train_params=train_params or {},
        random_state=random_state,
        pretrained_models=pretrained_models,
        pretrained_model=baseline_for_eval,
        model_type=model_type,
        experiment_type=experiment_type,
        fl_model_params=fl_model_params or {},
        test_loader=test_loader,
        test_df=test_df,
        test_input_cols=test_input_cols,
        test_target_col=test_target_col,
        baseline_original_model=baseline_original_model,
    )
    return result


def run_knowledge_distillation(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    input_cols: List[str],
    target_col: str,
    id_column: str,
    forget_ids: Iterable,
    device: Optional[str] = None,
    seq_len: int = 24,
    teacher_params: Dict = None,
    student_params: Dict = None,
    train_params: Dict = None,
    pretrained_teacher=None,
    baseline_model=None,
    model_type: str = "lstm",
    experiment_type: str = None,
    fl_model_params: Dict = None,
    test_loader: Optional[DataLoader] = None,
    test_df: Optional[pd.DataFrame] = None,
    test_input_cols: Optional[List[str]] = None,
    test_target_col: Optional[str] = None,
    baseline_original_model=None,
    use_federaser: bool = False,
    results_dir: Optional[str] = None,
    num_clients: Optional[int] = None,
    current_round: Optional[int] = None,
) -> Dict:
    device_t = _device_from_arg(device)
    #prefer the exact-retraining baseline for evaluation, else the pretrained teacher
    baseline_for_eval = baseline_model if baseline_model is not None else pretrained_teacher
    result = knowledge_distillation(
        X=X,
        y=y,
        df=df,
        input_cols=input_cols,
        target_col=target_col,
        id_column=id_column,
        forget_ids=forget_ids,
        seq_len=seq_len,
        device=device_t,
        teacher_params=teacher_params or {},
        student_params=student_params or {},
        train_params=train_params or {},
        pretrained_teacher=pretrained_teacher,
        baseline_model=baseline_for_eval,
        baseline_original_model=baseline_original_model,
        model_type=model_type,
        experiment_type=experiment_type,
        fl_model_params=fl_model_params or {},
        test_loader=test_loader,
        test_df=test_df,
        test_input_cols=test_input_cols,
        test_target_col=test_target_col,
        use_federaser=use_federaser,
        results_dir=results_dir,
        num_clients=num_clients,
        current_round=current_round,
    )
    return result
