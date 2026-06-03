"""Training functionality for federated learning client."""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def train_model(client, epochs):
    """Train the client's model for a number of epochs and return a results dict."""
    model = client.model
    train_loader = client.train_loader
    valid_loader = client.valid_loader
    device = client.device
    learning_rate = client.learning_rate
    client_id = client.client_id
    experiment_type = client.experiment_type

    # Set criterion based on experiment type
    if experiment_type == "n_cmapss":
        criterion = nn.MSELoss()
    elif experiment_type in ("mnist", "cifar10"):
        criterion = nn.CrossEntropyLoss()
    elif experiment_type.startswith("custom"):
        # Custom dataset: determine classification vs regression from labels
        all_labels = []
        for _, (_, target) in enumerate(train_loader):
            if target.dtype != torch.long:
                target = target.long()
            all_labels.extend(target.cpu().numpy())
        
        if len(all_labels) > 0:
            unique_labels = np.unique(all_labels)
            # If <= 10 unique integer labels, treat as classification
            is_classification = len(unique_labels) <= 10 and all(isinstance(l, (int, np.integer)) for l in unique_labels)
        else:
            is_classification = True  #Default
        
        if is_classification:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
    elif experiment_type in ("tabular", "adult"):
        #weighted CrossEntropy to handle class imbalance
        try:
            all_labels = []
            for _, (_, target) in enumerate(train_loader):
                if target.dtype != torch.long:
                    target = target.long()
                all_labels.extend(target.cpu().numpy())

            if len(all_labels) > 0:
                all_labels_array = np.array(all_labels)
                unique_labels, counts = np.unique(all_labels_array, return_counts=True)
                # number of classes the model outputs (read from its last linear layer)
                if hasattr(model, 'net') and len(list(model.net)) > 0:
                    last_layer = list(model.net)[-1]
                    if isinstance(last_layer, torch.nn.Linear):
                        num_classes = last_layer.out_features
                    else:
                        num_classes = int(all_labels_array.max()) + 1
                else:
                    num_classes = int(all_labels_array.max()) + 1

                num_classes_in_data = len(unique_labels)
                class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)

                # inverse-frequency weights: rare classes get more weight
                total = len(all_labels)
                for label, count in zip(unique_labels, counts):
                    if count > 0 and int(label) < num_classes:
                        class_weights[int(label)] = total / (num_classes_in_data * count)

                # normalize over classes actually present
                present_weights = class_weights[unique_labels]
                if present_weights.sum() > 0:
                    class_weights[unique_labels] = present_weights / present_weights.sum() * num_classes_in_data
                
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                print(f"Client {client_id}: Using weighted CrossEntropyLoss for {num_classes_in_data} classes (model expects {num_classes})")
            else:
                criterion = nn.CrossEntropyLoss()
        except Exception as e:
            print(f"Warning: Could not calculate class weights: {e}. Using standard CrossEntropyLoss.")
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    epoch_times = []
    batch_times = []

    #Advanced metrics tracking
    learning_stats = {
        "per_epoch_metrics": []
    }

    #Classification specific metrics
    if experiment_type == "mnist":
        learning_stats["classification_metrics"] = {
            "precision": [],
            "recall": [],
            "f1_score": [],
        }

    # Regression specific metrics
    elif experiment_type == "n_cmapss":
        learning_stats["regression_metrics"] = {
            "mae": [],  # Mean Absolute Error
            "mse": [],  # Mean Squared Error
            "r_squared": [],  #Coefficient of determination
        }

    model.train()
    print(f"Client {client_id} starting training for {epochs} epochs")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_batch_times = []

        #Training
        train_loss = 0
        model.train()
        correct = 0
        total = 0

        # For detailed metrics
        all_targets = []
        all_predictions = []

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()

            data, target = data.to(device), target.to(device)
            if experiment_type in ["mnist", "cifar10", "tabular"] or experiment_type.startswith("custom"):
                if target.dtype != torch.long and criterion.__class__.__name__ == 'CrossEntropyLoss':
                    target = target.long()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Store batch time
            batch_time = time.time() - batch_start_time
            epoch_batch_times.append(batch_time)

            # Calculate accuracy for classification tasks (MNIST and tabular)
            if experiment_type in ["mnist", "cifar10", "tabular"]:
                if target.dtype != torch.long:
                    target = target.long()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                #Store predictions and targets for detailed metrics
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            #For N-CMAPSS, collect predictions and targets
            elif experiment_type == "n_cmapss":
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(output.cpu().detach().numpy())

        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Calculate training accuracy for classification tasks (MNIST and tabular)
        train_acc = correct / total if experiment_type in ["mnist", "cifar10", "tabular"] and total > 0 else None
        if train_acc is not None:
            train_accuracies.append(train_acc)

        # Validation
        valid_loss = 0
        model.eval()
        val_correct = 0
        val_total = 0

        #For detailed validation metrics
        val_all_targets = []
        val_all_predictions = []

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                if experiment_type in ["mnist", "cifar10", "tabular"] and target.dtype != torch.long:
                    target = target.long()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()

                #Calculate accuracy for classification tasks (MNIST and tabular)
                if experiment_type in ["mnist", "cifar10", "tabular"]:
                    if target.dtype != torch.long:
                        target = target.long()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()

                    # Store predictions and targets for detailed metrics
                    val_all_targets.extend(target.cpu().numpy())
                    val_all_predictions.extend(predicted.cpu().numpy())

                # For N-CMAPSS, collect predictions and targets
                elif experiment_type == "n_cmapss":
                    val_all_targets.extend(target.cpu().numpy())
                    val_all_predictions.extend(output.cpu().detach().numpy())

        # Calculate average validation loss
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        #Calculate validation accuracy for classification tasks (MNIST and tabular)
        valid_acc = val_correct / val_total if experiment_type in ["mnist", "cifar10", "tabular"] and val_total > 0 else None
        if valid_acc is not None:
            valid_accuracies.append(valid_acc)

        #Calculate additional metrics based on experiment type
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }

        # Add classification metrics
        if experiment_type == "mnist":
            epoch_metrics["train_accuracy"] = train_acc
            epoch_metrics["valid_accuracy"] = valid_acc

            # Calculate advanced classification metrics on validation data
            if len(val_all_targets) > 0:
                try:
                    val_precision = precision_score(val_all_targets, val_all_predictions, average='macro', zero_division=0)
                    val_recall = recall_score(val_all_targets, val_all_predictions, average='macro', zero_division=0)
                    val_f1 = f1_score(val_all_targets, val_all_predictions, average='macro', zero_division=0)

                    # Record detailed metrics
                    epoch_metrics["valid_precision"] = val_precision
                    epoch_metrics["valid_recall"] = val_recall
                    epoch_metrics["valid_f1"] = val_f1

                    #Store in the classification metrics history
                    learning_stats["classification_metrics"]["precision"].append(val_precision)
                    learning_stats["classification_metrics"]["recall"].append(val_recall)
                    learning_stats["classification_metrics"]["f1_score"].append(val_f1)

                    #Compute confusion matrix (for the last epoch only to save space)
                    if epoch == epochs - 1:
                        cm = confusion_matrix(val_all_targets, val_all_predictions)
                        epoch_metrics["confusion_matrix"] = cm.tolist()
                except Exception as e:
                    print(f"Error calculating classification metrics: {e}")

        # Add regression metrics for N-CMAPSS
        elif experiment_type == "n_cmapss":
            # Calculate advanced regression metrics
            if len(val_all_targets) > 0:
                try:
                    val_targets = np.array(val_all_targets)
                    val_preds = np.array(val_all_predictions)

                    # Mean Absolute Error
                    val_mae = np.mean(np.abs(val_preds - val_targets))

                    #Mean Squared Error (already calculated as valid_loss)
                    val_mse = valid_loss

                    #R² (coefficient of determination)
                    val_mean_target = np.mean(val_targets)
                    val_ss_total = np.sum((val_targets - val_mean_target) ** 2)
                    val_ss_residual = np.sum((val_targets - val_preds) ** 2)
                    val_r_squared = 1 - (val_ss_residual / val_ss_total) if val_ss_total != 0 else 0

                    # Calculate % of predictions within ±10 and ±20 units
                    within_10 = np.mean(np.abs(val_preds - val_targets) <= 10.0) * 100
                    within_20 = np.mean(np.abs(val_preds - val_targets) <= 20.0) * 100

                    # Record detailed metrics
                    epoch_metrics["valid_mae"] = val_mae
                    epoch_metrics["valid_mse"] = val_mse
                    epoch_metrics["valid_r_squared"] = val_r_squared
                    epoch_metrics["valid_within_10"] = within_10
                    epoch_metrics["valid_within_20"] = within_20

                    # Store in the regression metrics history
                    learning_stats["regression_metrics"]["mae"].append(val_mae)
                    learning_stats["regression_metrics"]["mse"].append(val_mse)
                    learning_stats["regression_metrics"]["r_squared"].append(val_r_squared)
                except Exception as e:
                    print(f"Error calculating regression metrics: {e}")

        #Record timing information
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        batch_times.extend(epoch_batch_times)

        epoch_metrics["epoch_time_seconds"] = epoch_time
        epoch_metrics["avg_batch_time_seconds"] = np.mean(epoch_batch_times) if epoch_batch_times else 0

        #Add epoch metrics to the learning stats
        learning_stats["per_epoch_metrics"].append(epoch_metrics)

        # Print progress
        if experiment_type in ["mnist", "cifar10", "tabular"]:
            train_acc_str = f"{train_acc:.4f}" if train_acc is not None else "N/A"
            valid_acc_str = f"{valid_acc:.4f}" if valid_acc is not None else "N/A"
            print(f"Client {client_id} - Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc_str}, "
                  f"Valid Loss: {valid_loss:.6f}, Valid Acc: {valid_acc_str}")
        else:
            print(f"Client {client_id} - Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")

    # Create results dictionary
    training_results = {
        "client_id": client_id,
        "experiment_type": experiment_type,
        "epochs": epochs,
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "final_train_loss": train_losses[-1],
        "final_valid_loss": valid_losses[-1],
        "training_time": {
            "total_seconds": sum(epoch_times),
            "avg_epoch_seconds": np.mean(epoch_times),
            "avg_batch_seconds": np.mean(batch_times),
            "per_epoch_seconds": epoch_times
        },
        "learning_stats": learning_stats
    }

    # Add accuracy metrics for classification tasks (MNIST and tabular)
    if experiment_type in ["mnist", "cifar10", "tabular"]:
        if train_accuracies:
            training_results.update({
                "train_accuracies": train_accuracies,
                "final_train_accuracy": train_accuracies[-1],
                "accuracy": train_accuracies[-1]  #Add 'accuracy' key for compatibility
            })
        if valid_accuracies:
            training_results.update({
                "valid_accuracies": valid_accuracies,
                "final_valid_accuracy": valid_accuracies[-1]
            })
        #Fallback: use validation accuracy if available, otherwise training accuracy
        if not training_results.get("accuracy") and valid_accuracies:
            training_results["accuracy"] = valid_accuracies[-1]
        elif not training_results.get("accuracy") and train_accuracies:
            training_results["accuracy"] = train_accuracies[-1]

    print(f"Client {client_id} finished training")
    return training_results
