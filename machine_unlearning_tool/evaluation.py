from typing import Dict, Optional, List
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import linear_model, model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .model_utils import is_pytorch_model, is_sklearn_model


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)


@torch.no_grad()
def evaluate_model(net: torch.nn.Module, loader: DataLoader, device, is_classification: bool = False) -> Dict[str, float]:
    net.eval()
    ys = []
    yhats = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = net(xb)
        ys.append(yb.detach().cpu().numpy())
        yhats.append(preds.detach().cpu().numpy())
    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.float32)
    y_pred = np.concatenate(yhats) if yhats else np.array([], dtype=np.float32)
    
    if len(y_true) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    if is_classification:
        # logits -> predicted class
        if y_pred.ndim > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred
        accuracy = np.mean(y_pred_classes == y_true)
        return {"accuracy": float(accuracy)}
    else:
        # match shapes for regression metrics
        if y_pred.ndim > 1:
            y_pred = y_pred.squeeze()
        if y_true.ndim > 1:
            y_true = y_true.squeeze()
        return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def evaluate_model_universal(
    model,
    loader=None,
    X=None,
    y=None,
    device=None,
    is_classification: bool = False,
) -> Dict[str, float]:
    """Evaluate a PyTorch or sklearn model; returns rmse/mae/r2 or accuracy."""
    if is_pytorch_model(model):
        if loader is None:
            raise ValueError("PyTorch models require DataLoader")
        return evaluate_model(model, loader, device, is_classification=is_classification)
    elif is_sklearn_model(model) or hasattr(model, "predict"):
        if X is None or y is None:
            raise ValueError("sklearn-like models require X and y arrays")
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        y_pred = model.predict(X_flat)
        if is_classification:
            from sklearn.metrics import accuracy_score
            return {"accuracy": accuracy_score(y, y_pred)}
        else:
            return {
                "rmse": rmse(y, y_pred),
                "mae": mae(y, y_pred),
                "r2": r2(y, y_pred)
            }
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


@torch.no_grad()
def get_model_logits(model, loader: DataLoader, device) -> np.ndarray:
    """Haalt logits op (voor softmax) van een model."""
    model.eval()
    logits_list = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        logits_list.append(logits.detach().cpu().numpy())
    return np.concatenate(logits_list, axis=0) if logits_list else np.array([], dtype=np.float32)


@torch.no_grad()
def get_model_probs(model, loader: DataLoader, device) -> np.ndarray:
    """Haalt softmax probabilities op van een model."""
    model.eval()
    probs_list = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        probs_list.append(probs.detach().cpu().numpy())
    return np.concatenate(probs_list, axis=0) if probs_list else np.array([], dtype=np.float32)


def calculate_logit_mse(logits_unlearned: np.ndarray, logits_exact_retrain: np.ndarray) -> float:
    """Berekent MSE tussen logits van twee modellen. Lager = beter."""
    if logits_unlearned.shape != logits_exact_retrain.shape:
        raise ValueError(f"Shape mismatch: {logits_unlearned.shape} vs {logits_exact_retrain.shape}")

    if len(logits_unlearned) == 0:
        return float('nan')

    N, K = logits_unlearned.shape

    diff = logits_unlearned - logits_exact_retrain
    squared_norms = np.sum(diff ** 2, axis=1)

    # mean over samples, normalised by number of classes
    mse = np.mean(squared_norms) / K

    return float(mse)


def calculate_kl_divergence(probs_exact_retrain: np.ndarray, probs_unlearned: np.ndarray, epsilon: float = 1e-10) -> float:
    """Berekent KL-divergence tussen probability distributions. Lager = beter."""
    if probs_exact_retrain.shape != probs_unlearned.shape:
        raise ValueError(f"Shape mismatch: {probs_exact_retrain.shape} vs {probs_unlearned.shape}")
    
    if len(probs_exact_retrain) == 0:
        return float('nan')

    #clip to avoid log(0)
    probs_rt = np.clip(probs_exact_retrain, epsilon, 1.0)
    probs_unl = np.clip(probs_unlearned, epsilon, 1.0)

    kl_per_sample = np.sum(probs_rt * np.log(probs_rt / probs_unl), axis=1)
    return float(np.mean(kl_per_sample))


def calculate_behavioral_distance(
    model_unlearned,
    model_baseline,
    loader: DataLoader,
    device,
    is_classification: bool = True
) -> Dict[str, float]:
    """How differently the unlearned model behaves vs a baseline.

    Classification: logit-MSE + KL-divergence. Regression: MSE on outputs.
    """
    if not is_pytorch_model(model_unlearned) or not is_pytorch_model(model_baseline):
        return {}  #sklearn models have no logits

    if not is_classification:
        # regression: MSE between outputs
        outputs_unl = []
        outputs_baseline = []
        model_unlearned.eval()
        model_baseline.eval()
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                out_unl = model_unlearned(xb)
                out_baseline = model_baseline(xb)
                outputs_unl.append(out_unl.detach().cpu().numpy())
                outputs_baseline.append(out_baseline.detach().cpu().numpy())
        
        outputs_unl = np.concatenate(outputs_unl, axis=0) if outputs_unl else np.array([], dtype=np.float32)
        outputs_baseline = np.concatenate(outputs_baseline, axis=0) if outputs_baseline else np.array([], dtype=np.float32)

        if len(outputs_unl) == 0:
            return {}

        mse = np.mean((outputs_unl - outputs_baseline) ** 2)
        return {"output_mse": float(mse)}

    # classification: logit-MSE and KL-divergence
    logits_unl = get_model_logits(model_unlearned, loader, device)
    logits_baseline = get_model_logits(model_baseline, loader, device)

    if len(logits_unl) == 0:
        return {}

    logit_mse = calculate_logit_mse(logits_unl, logits_baseline)

    probs_unl = get_model_probs(model_unlearned, loader, device)
    probs_baseline = get_model_probs(model_baseline, loader, device)
    kl_div = calculate_kl_divergence(probs_baseline, probs_unl)

    return {
        "logit_mse": logit_mse,
        "kl_divergence": kl_div
    }


@torch.no_grad()
def compute_per_sample_losses(
    model,
    loader: DataLoader,
    device,
    is_classification: bool = False
) -> np.ndarray:
    """Per-sample loss for a PyTorch model (CrossEntropy or MSE). Used for MIA."""
    model.eval()
    losses = []

    if is_classification:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = torch.nn.MSELoss(reduction='none')

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)

        if is_classification:
            if yb.dtype != torch.long:
                yb = yb.long()
            loss = criterion(preds, yb)
        else:
            if preds.dim() > 1:
                preds = preds.squeeze()
            if yb.dim() > 1:
                yb = yb.squeeze()
            loss = criterion(preds, yb)

        losses.append(loss.detach().cpu().numpy())

    if not losses:
        return np.array([], dtype=np.float32)

    return np.concatenate(losses, axis=0)


def compute_per_sample_losses_sklearn(
    model,
    X: np.ndarray,
    y: np.ndarray,
    is_classification: bool = False
) -> np.ndarray:
    """Per-sample loss for an sklearn model (cross-entropy or MSE). Used for MIA."""
    if X.ndim == 3:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X

    y_pred = model.predict(X_flat)

    if is_classification:
        # need class probabilities; one-hot the predictions if not available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_flat)
        else:
            num_classes = len(np.unique(y))
            y_proba = np.zeros((len(y), num_classes))
            y_proba[np.arange(len(y)), y_pred] = 1.0

        #cross-entropy per sample: -log(P(true class))
        losses = []
        for i in range(len(y)):
            true_class = int(y[i])
            prob = np.clip(y_proba[i, true_class], 1e-10, 1.0)
            losses.append(-np.log(prob))
        return np.array(losses, dtype=np.float32)
    else:
        losses = (y - y_pred) ** 2
        return losses.astype(np.float32)


def simple_mia(
    sample_losses: np.ndarray,
    membership_labels: np.ndarray,
    n_splits: int = 10,
    random_state: int = 0,
    use_holdout: bool = True
) -> Dict[str, float]:
    """Membership Inference Attack on the per-sample losses.

    If unlearning works, an attacker can't tell forget samples (label 1, were in
    training) from test samples (label 0), so accuracy should be ~0.5.
    """
    if len(sample_losses) == 0:
        return {
            "mia_accuracy": float('nan'),
        }

    unique_labels = np.unique(membership_labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(f"membership_labels moet alleen 0 en 1 bevatten, kreeg: {unique_labels}")

    #drop non-finite losses, they break the LR fit
    sample_losses = np.asarray(sample_losses, dtype=np.float64)
    membership_labels = np.asarray(membership_labels)

    finite_mask = np.isfinite(sample_losses)
    if not finite_mask.any():
        return {
            "mia_accuracy": float('nan'),
        }
    sample_losses = sample_losses[finite_mask]
    membership_labels = membership_labels[finite_mask]

    # clip outliers via median/MAD to keep gradients stable
    median = np.median(sample_losses)
    mad = np.median(np.abs(sample_losses - median)) + 1e-12
    lower = median - 10 * mad
    upper = median + 10 * mad
    sample_losses = np.clip(sample_losses, lower, upper)
    sample_losses = np.clip(sample_losses, -1e10, 1e10)

    X_mia = sample_losses.reshape(-1, 1)
    y_mia = membership_labels

    # attack model: scaled logistic regression
    attack_model = make_pipeline(
        StandardScaler(),
        linear_model.LogisticRegression(random_state=random_state, max_iter=1000, solver="lbfgs")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if use_holdout:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X_mia, y_mia,
                test_size=0.3,
                random_state=random_state,
                stratify=y_mia
            )
            attack_model.fit(X_train, y_train)
            accuracy = attack_model.score(X_test, y_test)
            return {
                "mia_accuracy": float(accuracy),
            }
        else:
            cv = model_selection.StratifiedShuffleSplit(
                n_splits=n_splits,
                random_state=random_state,
                test_size=0.2
            )
            scores = model_selection.cross_val_score(
                attack_model, X_mia, y_mia, cv=cv, scoring='accuracy'
            )
            return {
                "mia_accuracy": float(np.mean(scores)),
            }


@torch.no_grad()
def compute_activation_distance(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    data_loader: DataLoader,
    device,
    layer_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Cosine similarity between two models' hidden activations.

    Compares internal features rather than outputs: low similarity means the
    unlearned model represents the data differently, a sign of good unlearning.
    """
    if not is_pytorch_model(model1) or not is_pytorch_model(model2):
        return {}

    model1.eval()
    model2.eval()

    # pick layers: common names first, else any Linear layer
    if layer_names is None:
        model1_layers = dict(model1.named_modules())
        possible_names = ['fc1', 'fc2', 'linear1', 'linear2', 'hidden', 'layer1', 'layer2']
        layer_names = [name for name in possible_names if name in model1_layers]

        if not layer_names:
            layer_names = [name for name, module in model1_layers.items()
                          if isinstance(module, torch.nn.Linear) and name != '']

        if not layer_names:
            return {}

    results = {}

    for layer_name in layer_names[:1]:  #only the first usable layer
        try:
            activations1 = []
            activations2 = []

            #forward hook that stores the layer output
            def get_activation(storage):
                def hook(model, input, output):
                    storage.append(output.detach().cpu().numpy())
                return hook

            layer1 = dict(model1.named_modules())[layer_name]
            layer2 = dict(model2.named_modules())[layer_name]

            hook1 = layer1.register_forward_hook(get_activation(activations1))
            hook2 = layer2.register_forward_hook(get_activation(activations2))

            for X, _ in data_loader:
                X = X.to(device)
                model1(X)
                model2(X)

            hook1.remove()
            hook2.remove()

            if not activations1 or not activations2:
                continue

            act1 = np.concatenate(activations1, axis=0)
            act2 = np.concatenate(activations2, axis=0)

            if act1.ndim > 2:
                act1 = act1.reshape(act1.shape[0], -1)
            if act2.ndim > 2:
                act2 = act2.reshape(act2.shape[0], -1)

            # cosine similarity per sample: (a·b) / (||a|| ||b||)
            dot_products = (act1 * act2).sum(axis=1)
            norms1 = np.linalg.norm(act1, axis=1)
            norms2 = np.linalg.norm(act2, axis=1)

            norms_product = norms1 * norms2
            norms_product = np.where(norms_product > 0, norms_product, 1.0)  # avoid /0
            cosine_similarities = dot_products / norms_product

            results = {
                "activation_cosine_similarity": float(cosine_similarities.mean()),
            }
            break

        except (KeyError, RuntimeError):
            continue  # try the next layer

    return results


@torch.no_grad()
def compute_confidence_metrics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device
) -> Dict[str, float]:
    """Mean confidence and entropy on the given data (low confidence + high entropy = good after unlearning)."""
    if not is_pytorch_model(model):
        return {}

    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    if not all_probs:
        return {}

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    confidence = probs.max(axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)  #H = -sum(p log p)

    correct_mask = (preds == labels)
    confidence_correct = confidence[correct_mask].mean() if correct_mask.sum() > 0 else 0.0
    confidence_incorrect = confidence[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0.0

    return {
        "confidence_mean": float(confidence.mean()),
        "entropy_mean": float(entropy.mean()),
    }


@torch.no_grad()
def compute_js_divergence(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    data_loader: DataLoader,
    device
) -> Dict[str, float]:
    """JS divergence between the two models' predictions (symmetric, in [0, 1])."""
    if not is_pytorch_model(model1) or not is_pytorch_model(model2):
        return {}

    model1.eval()
    model2.eval()

    all_probs1 = []
    all_probs2 = []

    for X, _ in data_loader:
        X = X.to(device)

        logits1 = model1(X)
        logits2 = model2(X)

        probs1 = F.softmax(logits1, dim=1).cpu().numpy()
        probs2 = F.softmax(logits2, dim=1).cpu().numpy()

        all_probs1.append(probs1)
        all_probs2.append(probs2)

    if not all_probs1:
        return {}

    probs1 = np.concatenate(all_probs1, axis=0)
    probs2 = np.concatenate(all_probs2, axis=0)

    #JS = 0.5*KL(P||M) + 0.5*KL(Q||M), with M the mean distribution
    m = 0.5 * (probs1 + probs2)

    epsilon = 1e-10  # avoid log(0)
    probs1 = np.clip(probs1, epsilon, 1.0)
    probs2 = np.clip(probs2, epsilon, 1.0)
    m = np.clip(m, epsilon, 1.0)

    kl1 = np.sum(probs1 * np.log(probs1 / m), axis=1)
    kl2 = np.sum(probs2 * np.log(probs2 / m), axis=1)
    js_div = 0.5 * (kl1 + kl2)

    return {
        "js_divergence_mean": float(js_div.mean()),
        "js_divergence_max": float(js_div.max())
    }
