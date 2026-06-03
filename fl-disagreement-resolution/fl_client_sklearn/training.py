"""Sklearn model training for federated learning client."""

import time
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score


def train_sklearn_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    experiment_type: str = "n_cmapss"
) -> Dict[str, Any]:
    """Train sklearn model on client data.
    
    Unlike PyTorch, sklearn models don't need epochs - they just fit once.
    
    Args:
        model: Sklearn model to train
        X_train: Training features
        y_train: Training labels
        experiment_type: Type of experiment ('n_cmapss' or 'mnist')
    
    Returns:
        Dictionary containing training results
    """
    training_start_time = time.time()
    
    # Train the model (sklearn just needs fit(), no epochs)
    model.fit(X_train, y_train)
    
    training_time = time.time() - training_start_time
    
    #Evaluate on training data
    y_pred = model.predict(X_train)
    
    if experiment_type == "n_cmapss":
        #Regression metrics
        train_mse = mean_squared_error(y_train, y_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, y_pred)
        train_r2 = r2_score(y_train, y_pred)
        
        metrics = {
            "train_loss": float(train_rmse),
            "train_mse": float(train_mse),
            "train_mae": float(train_mae),
            "train_r2": float(train_r2)
        }
    
    elif experiment_type == "mnist":
        # Classification metrics
        train_accuracy = accuracy_score(y_train, y_pred)
        
        metrics = {
            "train_loss": 1.0 - float(train_accuracy),
            "train_accuracy": float(train_accuracy)
        }
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    training_results = {
        "training_time": {
            "total_seconds": training_time
        },
        "metrics": metrics,
        "num_samples": len(X_train)
    }
    
    return training_results

