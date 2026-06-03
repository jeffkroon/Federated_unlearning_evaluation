"""Sklearn model evaluation for federated learning."""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report


def evaluate_sklearn_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    experiment_type: str = "n_cmapss"
) -> Dict[str, float]:
    """Evaluate sklearn model on test data.
    
    Args:
        model: Sklearn model or ensemble
        X_test: Test features
        y_test: Test labels
        experiment_type: Type of experiment ('n_cmapss' or 'mnist')
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Check if model is fitted (for sklearn models)
    # For our custom ensemble wrappers, check if estimators are fitted
    try:
        from sklearn.utils.validation import check_is_fitted
        from fl_server_sklearn.aggregation import _WeightedVotingClassifier, _WeightedVotingRegressor
        
        # For our custom ensembles, check if underlying estimators are fitted
        if isinstance(model, (_WeightedVotingClassifier, _WeightedVotingRegressor)):
            #Check if any estimator is fitted
            has_fitted = False
            for name, est in model.estimators:
                try:
                    check_is_fitted(est)
                    has_fitted = True
                    break
                except:
                    continue
            if not has_fitted:
                raise ValueError("No fitted estimators in ensemble")
        else:
            #For regular models, check directly
            check_is_fitted(model)
    except Exception as e:
        # Model not fitted yet (e.g., initial model before training)
        print(f"Warning: Model not fitted yet. Skipping evaluation. ({e})")
        return {
            "loss": float('inf'),
            "accuracy": 0.0 if experiment_type == "mnist" else None,
            "note": "Model not fitted"
        }
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    if experiment_type == "n_cmapss":
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "loss": float(rmse)  #Use RMSE as loss
        }
    
    elif experiment_type == "mnist":
        #Classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # For classification, we can also compute per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return {
            "accuracy": float(accuracy),
            "loss": 1.0 - float(accuracy),  # Use 1 - accuracy as loss
            "classification_report": report
        }
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

