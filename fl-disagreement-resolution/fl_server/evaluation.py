"""Model evaluation functionality for federated learning."""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import copy
try:
    from brokenaxes import brokenaxes
except ImportError:
    brokenaxes = None  #Optional dependency for visualization

from fl_server.utils import make_json_serializable, read_client_results_from_files


def _is_classification_task(server):
    """Whether server task is classification (plug-and-play: use adapter or server._is_classification)."""
    if getattr(server, "_is_classification", None) is not None:
        return server._is_classification
    try:
        from fl_module.registry import DatasetAdapterRegistry
        adapter = DatasetAdapterRegistry.get_adapter(server.experiment_type)
        if adapter is not None:
            return adapter.is_classification()
    except Exception:
        pass
    return server.experiment_type in ("mnist", "cifar10", "tabular") or (
        isinstance(server.experiment_type, str) and server.experiment_type.startswith("custom")
    )


def evaluate_model(server, fl_round=None, client_results=None):
    """Evaluate the global model on the test set; returns (test_loss, accuracy-or-None)."""
    server.global_model.eval()

    if fl_round is None:
        fl_round = server.round
    else:
        server.round = fl_round

    # Read client results from filesystem if not round 0
    if fl_round > 0 and not client_results and server.results_dir and server.client_ids:
        client_results = read_client_results_from_files(server.results_dir, server.client_ids, fl_round)

    # Set criterion based on task type (plug-and-play)
    if _is_classification_task(server):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    test_loss = 0
    predictions = []
    actual = []
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in server.test_loader:
            data, target = data.to(server.device), target.to(server.device)
            output = server.global_model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            if _is_classification_task(server):
                if target.dtype != torch.long:
                    target = target.long()
                _, predicted = torch.max(output.data, 1)
                predictions.extend(predicted.cpu().numpy())
                actual.extend(target.cpu().numpy())
                total += target.size(0)
                correct += (predicted == target).sum().item()
            else:
                predictions.extend(output.cpu().numpy())
                actual.extend(target.cpu().numpy())

    # Calculate average test loss
    test_loss /= len(server.test_loader)

    #For regression, calculate RMSE and additional metrics
    if not _is_classification_task(server):
        rmse = np.sqrt(test_loss)
        test_loss = rmse  #RMSE as the primary test loss metric

        # Convert to numpy arrays for calculation
        predictions = np.array(predictions)
        actual = np.array(actual)

        # Calculate Mean Absolute Error
        mae = np.mean(np.abs(predictions - actual))

        # Calculate R-squared
        mean_actual = np.mean(actual)
        ss_total = np.sum((actual - mean_actual) ** 2)
        ss_residual = np.sum((actual - predictions) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        #Calculate % of predictions within 10 and 20 cycles
        within_10_cycles = np.mean(np.abs(predictions - actual) <= 10.0) * 100
        within_20_cycles = np.mean(np.abs(predictions - actual) <= 20.0) * 100

        print(f"Round {server.round} - RUL Prediction Metrics:")
        print(f"  RMSE: {rmse:.2f} cycles")
        print(f"  MAE: {mae:.2f} cycles")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Within ±10 cycles: {within_10_cycles:.2f}%")
        print(f"  Within ±20 cycles: {within_20_cycles:.2f}%")

        #Store additional metrics in training history
        if "rul_mae" not in server.training_history:
            server.training_history["rul_mae"] = []
        if "rul_r_squared" not in server.training_history:
            server.training_history["rul_r_squared"] = []
        if "rul_within_10" not in server.training_history:
            server.training_history["rul_within_10"] = []
        if "rul_within_20" not in server.training_history:
            server.training_history["rul_within_20"] = []

        server.training_history["rul_mae"].append(mae)
        server.training_history["rul_r_squared"].append(r_squared)
        server.training_history["rul_within_10"].append(within_10_cycles)
        server.training_history["rul_within_20"].append(within_20_cycles)

        # Update results dictionary
        round_results = {
            "round": fl_round,
            "test_loss": test_loss,
            "mae": mae,
            "r_squared": r_squared,
            "within_10_cycles": within_10_cycles,
            "within_20_cycles": within_20_cycles
        }

        # Add client results if provided
        if client_results:
            round_results["client_results"] = client_results

        # Add to results history
        server.results["rounds"].append(round_results)

        accuracy = None

    #For classification, calculate accuracy and other metrics
    elif _is_classification_task(server) or (isinstance(server.experiment_type, str) and server.experiment_type.startswith("custom")):
        accuracy = correct / total if total > 0 else 0.0

        #Calculate precision, recall, and F1 score for each class
        # detailed metrics only for non-custom experiments
        if (isinstance(server.experiment_type, str) and server.experiment_type.startswith("custom")):
            # For custom, use weighted average
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predictions, average='weighted', zero_division=0
            )
            per_class_metrics = {}  # Skip per-class for custom
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predictions, average=None, zero_division=0
            )

        #Calculate mean metrics
        mean_precision, mean_recall, mean_f1, _ = precision_recall_fscore_support(
            actual, predictions, average='weighted', zero_division=0
        )

        #Calculate per-class accuracy
        class_labels = np.unique(actual)
        per_class_accuracy = []
        for c in class_labels:
            mask = np.array(actual) == c
            class_acc = np.mean(np.array(predictions)[mask] == c) if np.sum(mask) > 0 else 0
            per_class_accuracy.append(class_acc)

        dataset_name = "MNIST" if server.experiment_type == "mnist" else ("CIFAR-10" if server.experiment_type == "cifar10" else "Tabular")
        print(f"Round {server.round} - {dataset_name} Classification Metrics:")
        print(f"  Overall Accuracy: {accuracy:.4f}")
        print(f"  Mean Precision: {mean_precision:.4f}")
        print(f"  Mean Recall: {mean_recall:.4f}")
        print(f"  Mean F1 Score: {mean_f1:.4f}")

        # Store additional metrics in training history
        if "mnist_precision" not in server.training_history:
            server.training_history["mnist_precision"] = []
        if "mnist_recall" not in server.training_history:
            server.training_history["mnist_recall"] = []
        if "mnist_f1" not in server.training_history:
            server.training_history["mnist_f1"] = []
        if "mnist_per_class_accuracy" not in server.training_history:
            server.training_history["mnist_per_class_accuracy"] = []

        server.training_history["mnist_precision"].append(mean_precision)
        server.training_history["mnist_recall"].append(mean_recall)
        server.training_history["mnist_f1"].append(mean_f1)
        server.training_history["mnist_per_class_accuracy"].append(per_class_accuracy)

        print(f"Round {server.round} - Global model test accuracy: {accuracy:.4f}")

        # Update results dictionary
        round_results = {
            "round": fl_round,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "mean_precision": mean_precision if 'mean_precision' in locals() else None,
            "mean_recall": mean_recall if 'mean_recall' in locals() else None,
            "mean_f1": mean_f1 if 'mean_f1' in locals() else None
        }

        # Add client results if provided
        if client_results:
            round_results["client_results"] = client_results

        #Add to results history
        server.results["rounds"].append(round_results)

    #Store history
    server.training_history["rounds"].append(server.round)
    server.training_history["global_test_loss"].append(test_loss)
    if accuracy is not None:
        server.training_history["global_test_accuracy"].append(accuracy)

    print(f"Round {server.round} - Global model test loss: {test_loss:.6f}")

    # Plot and save results
    save_evaluation_results(server, predictions, actual)

    # Evaluate each track model if this isn't round 0
    if fl_round > 0 and server.results_dir:
        print(f"\nEvaluating track models for round {fl_round}.")
        track_results = evaluate_track_models(server, fl_round)

        # Store track results in the main results dictionary
        if track_results:
            #Add track results to the round results
            for round_result in server.results["rounds"]:
                if round_result["round"] == fl_round:
                    round_result["track_results"] = track_results
                    break

            #If no track results storage found in training history, create it
            if "track_results" not in server.training_history:
                server.training_history["track_results"] = {}

            server.training_history["track_results"][str(fl_round)] = track_results

            print(f"\nTrack evaluation summary for round {fl_round}:")
            if accuracy is not None:
                print(f"Global model - Accuracy: {accuracy:.6f}")
            else:
                print(f"Global model - RMSE: {test_loss:.6f}")

            for track_name, track_data in track_results.items():
                if _is_classification_task(server):
                    print(f"{track_name} - Accuracy: {track_data['accuracy']:.6f}")
                else:
                    print(f"{track_name} - RMSE: {track_data['rmse']:.6f}")

            print()

    # Save experiment results
    server._save_experiment_results()

    return test_loss, accuracy

def save_evaluation_results(server, predictions, actual):
    """Save the training history JSON and (in verbose/last round) the plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine if this is the last round
    is_last_round = hasattr(server, 'fl_rounds') and server.round == server.fl_rounds

    # Determine output paths based on results_dir
    if server.results_dir:
        history_path = os.path.join(server.output_dir, f"training_history_round_{server.round}.json")
        loss_plot_path = os.path.join(server.output_dir, "plots", f"global_model_loss_round_{server.round}.png")

        if not _is_classification_task(server):
            pred_plot_path = os.path.join(server.output_dir, "plots", f"rul_prediction_round_{server.round}.png")
            metric_plot_path = os.path.join(server.output_dir, "plots", f"rul_metrics_round_{server.round}.png")
        else:
            cm_plot_path = os.path.join(server.output_dir, "plots", f"mnist_confusion_matrix_round_{server.round}.png")
            acc_plot_path = os.path.join(server.output_dir, "plots", f"global_model_accuracy_round_{server.round}.png")
    else:
        history_path = f"output/server_results/training_history_round_{server.round}_{timestamp}.json"
        loss_plot_path = f"output/plots/global_model_loss_round_{server.round}_{timestamp}.png"

        if not _is_classification_task(server):
            pred_plot_path = f"output/plots/rul_prediction_round_{server.round}_{timestamp}.png"
            metric_plot_path = f"output/plots/rul_metrics_round_{server.round}_{timestamp}.png"
        else:
            cm_plot_path = f"output/plots/mnist_confusion_matrix_round_{server.round}_{timestamp}.png"
            acc_plot_path = f"output/plots/global_model_accuracy_round_{server.round}_{timestamp}.png"

    #Convert numpy values to Python native types for JSON serialization
    history_for_json = make_json_serializable(server.training_history)

    #Save training history
    with open(history_path, "w") as f:
        json.dump(history_for_json, f)

    # Generate plots based on verbose_plots setting
    if server.verbose_plots or is_last_round:
        # Plot and save loss history
        plt.figure(figsize=(10, 6))
        plt.plot(server.training_history["rounds"], server.training_history["global_test_loss"], marker='o')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Test Loss')
        plt.title(f'Global Model Performance ({server.experiment_type})')
        plt.grid(True)
        plt.savefig(loss_plot_path)
        plt.close()

        if not _is_classification_task(server):
            plot_ncmapss_results(server, predictions, actual, pred_plot_path, metric_plot_path, server.verbose_plots or is_last_round)
        else:
            plot_mnist_results(server, predictions, actual, cm_plot_path, acc_plot_path, timestamp, server.verbose_plots or is_last_round)

    # plot track progress (needs >1 round)
    if server.round > 1 and "track_results" in server.training_history:
        plot_track_progress(server, server.round, server.verbose_plots or is_last_round)

    #timing metrics: verbose mode or last round only
    if (server.verbose_plots or is_last_round) and hasattr(server, 'aggregation_timing_history') and len(server.aggregation_timing_history) > 0:
        plot_timing_metrics(server, server.round)

    plot_mode = "verbose" if server.verbose_plots else ("last round" if is_last_round else "minimal")
    print(f"Saved results for round {server.round} (plot mode: {plot_mode})")

def plot_ncmapss_results(server, predictions, actual, pred_plot_path, metric_plot_path, should_plot=True):
    """Save the N-CMAPSS RUL scatter plot and per-round metric curves."""
    if not should_plot:
        return
    predictions = np.array(predictions)
    actual = np.array(actual)

    #Calculate error thresholds for coloring
    errors = predictions - actual
    within_10 = np.abs(errors) <= 10
    within_20 = np.logical_and(np.abs(errors) > 10, np.abs(errors) <= 20)
    beyond_20 = np.abs(errors) > 20

    # Create prediction scatter plot with colored points based on error
    plt.figure(figsize=(10, 6))

    # Plot points outside 20 cycles first (red)
    plt.scatter(actual[beyond_20], predictions[beyond_20], color='red', alpha=0.5, label='Error > 20 cycles')

    # Plot points within 10-20 cycles (yellow)
    plt.scatter(actual[within_20], predictions[within_20], color='orange', alpha=0.5, label='Error 10-20 cycles')

    #Plot points within 10 cycles last (green)
    plt.scatter(actual[within_10], predictions[within_10], color='green', alpha=0.5, label='Error ≤ 10 cycles')

    #Add perfect prediction line
    min_val = min(np.min(actual), np.min(predictions))
    max_val = max(np.max(actual), np.max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

    # Add ±10 cycle lines
    plt.plot([min_val, max_val], [min_val + 10, max_val + 10], 'g--', alpha=0.3)
    plt.plot([min_val, max_val], [min_val - 10, max_val - 10], 'g--', alpha=0.3)

    # Add ±20 cycle lines
    plt.plot([min_val, max_val], [min_val + 20, max_val + 20], 'orange', linestyle='--', alpha=0.3)
    plt.plot([min_val, max_val], [min_val - 20, max_val - 20], 'orange', linestyle='--', alpha=0.3)

    plt.xlabel('Actual RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')

    # Get current metrics
    rmse = server.training_history["global_test_loss"][-1]
    mae = server.training_history["rul_mae"][-1]
    r2 = server.training_history["rul_r_squared"][-1]
    within_10_pct = server.training_history["rul_within_10"][-1]
    within_20_pct = server.training_history["rul_within_20"][-1]

    plt.title(f'RUL Prediction - Round {server.round}\n'
             f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}\n'
             f'Within ±10 cycles: {within_10_pct:.2f}%, Within ±20 cycles: {within_20_pct:.2f}%')

    plt.legend()
    plt.grid(True)
    plt.savefig(pred_plot_path)
    plt.close()

    #plot extra metrics (needs >=2 rounds)
    if len(server.training_history["rounds"]) >= 2:
        plt.figure(figsize=(15, 10))

        #Create a 2x2 grid of subplots
        plt.subplot(2, 2, 1)
        plt.plot(server.training_history["rounds"], server.training_history["global_test_loss"], marker='o')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('RMSE (cycles)')
        plt.title('Root Mean Squared Error')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(server.training_history["rounds"], server.training_history["rul_mae"], marker='o', color='orange')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('MAE (cycles)')
        plt.title('Mean Absolute Error')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(server.training_history["rounds"], server.training_history["rul_within_10"], marker='o', color='green')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Percentage (%)')
        plt.title('Predictions Within ±10 Cycles')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(server.training_history["rounds"], server.training_history["rul_r_squared"], marker='o', color='purple')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('R²')
        plt.title('Coefficient of Determination (R²)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(metric_plot_path)
        plt.close()

def plot_mnist_results(server, predictions, actual, cm_plot_path, acc_plot_path, timestamp, should_plot=True):
    """Save the confusion matrix and per-round classification metric curves."""
    if not should_plot:
        return
    predictions = np.array(predictions)
    actual = np.array(actual)

    # Get the current metrics from the training history
    current_accuracy = server.training_history["global_test_accuracy"][-1]
    current_precision = server.training_history["mnist_precision"][-1]
    current_recall = server.training_history["mnist_recall"][-1]
    current_f1 = server.training_history["mnist_f1"][-1]
    current_per_class_acc = server.training_history["mnist_per_class_accuracy"][-1]

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(actual, predictions)
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #two subplots: raw + normalized confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    #Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted Labels')
    ax1.set_ylabel('True Labels')
    ax1.set_title('Confusion Matrix (Raw Counts)')

    # Normalized by row (true label)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted Labels')
    ax2.set_ylabel('True Labels')
    ax2.set_title('Confusion Matrix (Normalized by True Label)')

    plt.suptitle(f'Classification Results - Round {server.round}')
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()

    # plot metrics history (needs >=2 rounds)
    if len(server.training_history["rounds"]) >= 2:
        # Create a 2x2 subplot for accuracy, precision, recall, and F1 score
        plt.figure(figsize=(15, 10))

        #Accuracy
        plt.subplot(2, 2, 1)
        plt.plot(server.training_history["rounds"], server.training_history["global_test_accuracy"],
                 marker='o', color='blue')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Accuracy')
        plt.title('Overall Accuracy')
        plt.grid(True)

        #Precision
        plt.subplot(2, 2, 2)
        plt.plot(server.training_history["rounds"], server.training_history["mnist_precision"],
                 marker='o', color='green')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Precision')
        plt.title('Weighted Precision')
        plt.grid(True)

        # Recall
        plt.subplot(2, 2, 3)
        plt.plot(server.training_history["rounds"], server.training_history["mnist_recall"],
                 marker='o', color='orange')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Recall')
        plt.title('Weighted Recall')
        plt.grid(True)

        # F1 Score
        plt.subplot(2, 2, 4)
        plt.plot(server.training_history["rounds"], server.training_history["mnist_f1"],
                 marker='o', color='purple')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('F1 Score')
        plt.title('Weighted F1 Score')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(acc_plot_path)
        plt.close()

        # Plot per-class metrics for the current round
        #Get unique classes that actually appear in the data
        unique_classes = np.unique(np.concatenate([actual, predictions]))
        num_classes = len(unique_classes)
        
        #skip per-class plot for many-class tabular
        if server.experiment_type == "tabular" and num_classes > 50:
            print(f"Skipping per-class plot for tabular data with {num_classes} classes (too many to visualize)")
            return
        
        plt.figure(figsize=(min(20, num_classes * 0.5), 6))

        # Retrieve per-class metrics for the latest round
        # per-class precision, recall, F1
        precision_per_class, recall_per_class, f1_per_class, labels = precision_recall_fscore_support(
            actual, predictions, average=None, zero_division=0, labels=unique_classes
        )

        # Get per-class accuracy for present classes only
        per_class_acc_present = []
        for cls in unique_classes:
            mask = actual == cls
            if mask.sum() > 0:
                acc = (predictions[mask] == actual[mask]).sum() / mask.sum()
                per_class_acc_present.append(acc)
            else:
                per_class_acc_present.append(0.0)
        
        #Use only classes that appear in data
        classes = unique_classes
        x = np.arange(len(classes))
        width = 0.2

        #Bar chart with per-class metrics (only for classes present in data)
        plt.bar(x - 1.5*width, per_class_acc_present, width, label='Accuracy', color='blue')
        plt.bar(x - 0.5*width, precision_per_class, width, label='Precision', color='green')
        plt.bar(x + 0.5*width, recall_per_class, width, label='Recall', color='orange')
        plt.bar(x + 1.5*width, f1_per_class, width, label='F1 Score', color='purple')

        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.title(f'Per-class Metrics - Round {server.round}')
        plt.xticks(x, classes)
        plt.legend()
        plt.grid(True, axis='y')

        # Add the current round's metrics as a subtitle
        plt.figtext(0.5, 0.01,
                   f"Overall: Acc={current_accuracy:.4f}, Prec={current_precision:.4f}, Rec={current_recall:.4f}, F1={current_f1:.4f}",
                   ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

        if server.results_dir:
            per_class_path = os.path.join(server.output_dir, "plots", f"mnist_per_class_metrics_round_{server.round}.png")
        else:
            per_class_path = f"output/plots/mnist_per_class_metrics_round_{server.round}_{timestamp}.png"

        plt.tight_layout()
        plt.savefig(per_class_path)
        plt.close()

def evaluate_track_models(server, round_num):
    """Evaluate each track's model (post-unlearning branch if present) on the test set."""
    structure = server._get_structure_config()

    # Path to tracks directory
    tracks_dir = os.path.join(
        server.results_dir,
        structure["round_template"].format(round=round_num),
        "tracks"
    )

    # Check if tracks directory exists
    if not os.path.exists(tracks_dir):
        print(f"No tracks directory found for round {round_num}")

        #Check if there were tracks in previous rounds
        had_previous_tracks = False
        for prev_round in range(1, round_num):
            prev_tracks_dir = os.path.join(
                server.results_dir,
                structure["round_template"].format(round=prev_round),
                "tracks"
            )
            if os.path.exists(prev_tracks_dir):
                had_previous_tracks = True
                break

        #there were tracks before but not now, so the disagreements expired
        # Evaluate just the global model for comparison
        if had_previous_tracks:
            print(f"Disagreements have expired in round {round_num}, evaluating only the global model")

            # Set criterion based on experiment type
            if not _is_classification_task(server):
                criterion = nn.MSELoss()
            elif _is_classification_task(server):
                criterion = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown experiment type: {server.experiment_type}")

            # Path to global model for this round
            global_model_dir = os.path.join(
                server.results_dir,
                structure["round_template"].format(round=round_num),
                structure["global_model_aggregated"]
            )
            global_model_path = os.path.join(global_model_dir, "model.pt")

            if not os.path.exists(global_model_path):
                print(f"Global model file not found for round {round_num}")
                return {}

            #Save the current global model state
            original_state = copy.deepcopy(server.global_model.state_dict())

            #Load the baseline global model (reference) for evaluation
            # all-client FedAvg, no unlearning, comparison baseline
            server.global_model.load_state_dict(torch.load(global_model_path, map_location=server.device))
            server.global_model.eval()

            # Initialize metrics
            test_loss = 0
            predictions = []
            actual = []
            correct = 0
            total = 0

            # Evaluate the model
            with torch.no_grad():
                for data, target in server.test_loader:
                    data, target = data.to(server.device), target.to(server.device)
                    output = server.global_model(data)
                    loss = criterion(output, target)
                    test_loss += loss.item()

                    #For regression (N-CMAPSS)
                    if not _is_classification_task(server):
                        predictions.extend(output.cpu().numpy())
                        actual.extend(target.cpu().numpy())
                    #For classification (MNIST/CIFAR-10)
                    elif _is_classification_task(server):
                        _, predicted = torch.max(output.data, 1)
                        predictions.extend(predicted.cpu().numpy())
                        actual.extend(target.cpu().numpy())
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

            # Calculate average test loss
            test_loss /= len(server.test_loader)

            # Create results for baseline global model (reference)
            track_results = {}

            if not _is_classification_task(server):
                rmse = np.sqrt(test_loss)
                predictions = np.array(predictions)
                actual = np.array(actual)
                mae = np.mean(np.abs(predictions - actual))
                mean_actual = np.mean(actual)
                ss_total = np.sum((actual - mean_actual) ** 2)
                ss_residual = np.sum((actual - predictions) ** 2)
                r_squared = 1 - (ss_residual / ss_total)
                within_10_cycles = np.mean(np.abs(predictions - actual) <= 10.0) * 100
                within_20_cycles = np.mean(np.abs(predictions - actual) <= 20.0) * 100

                track_results["baseline_global"] = {
                    "rmse": rmse,
                    "mae": mae,
                    "r_squared": r_squared,
                    "within_10_cycles": within_10_cycles,
                    "within_20_cycles": within_20_cycles,
                    "model_source": "global_baseline"
                }

                print(f"Baseline global model (reference) - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r_squared:.4f}")

            elif _is_classification_task(server):
                accuracy = correct / total if total > 0 else 0
                precision, recall, f1, _ = precision_recall_fscore_support(
                    actual, predictions, average='weighted', zero_division=0
                )

                track_results["baseline_global"] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "test_loss": test_loss,
                    "model_source": "global_baseline"
                }

                print(f"Baseline global model (reference) - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

            # Restore the original global model state
            server.global_model.load_state_dict(original_state)

            #Save track results to a file
            if track_results and server.results_dir:
                results_path = os.path.join(
                    server.output_dir,
                    f"track_evaluation_round_{round_num}.json"
                )

                with open(results_path, 'w') as f:
                    json.dump(make_json_serializable(track_results), f, indent=2)

            return track_results

        return {}

    #Helper to handle SISA ensembles saved as ModuleList
    def _build_sisa_ensemble(base_model, state_dict):
        """Create an ensemble wrapper matching state_dict keys like models.0.*"""
        import torch.nn as nn
        import copy
        # Infer number of submodels from state_dict keys
        indices = []
        for k in state_dict.keys():
            parts = k.split(".")
            if len(parts) > 2 and parts[0] == "models":
                try:
                    indices.append(int(parts[1]))
                except ValueError:
                    continue
        num_models = max(indices) + 1 if indices else 1

        class SISAEnsemble(nn.Module):
            def __init__(self, prototype, n):
                super().__init__()
                self.models = nn.ModuleList([copy.deepcopy(prototype) for _ in range(n)])

            def forward(self, x):
                outputs = [m(x) for m in self.models]
                return torch.mean(torch.stack(outputs, dim=0), dim=0)

        return SISAEnsemble(base_model, num_models)

    # Get track metadata
    metadata_path = os.path.join(tracks_dir, "track_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"No track metadata found for round {round_num}")
        return {}

    try:
        with open(metadata_path, 'r') as f:
            track_metadata = json.load(f)

        track_names = list(track_metadata.get("tracks", {}).keys())
        print(f"Found {len(track_names)} tracks to evaluate: {track_names}")

        # Initialize results
        track_results = {}

        #Set criterion based on experiment type
        if not _is_classification_task(server):
            criterion = nn.MSELoss()
        elif _is_classification_task(server):
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown experiment type: {server.experiment_type}")

        #Save the current global model state/object
        original_model_obj = server.global_model
        original_state = copy.deepcopy(server.global_model.state_dict())

        # Evaluate each track model
        # tracks = client groups by disagreement pattern
        # Each track may have unlearning applied if it excludes clients
        for track_name in track_names:
            track_dir = os.path.join(tracks_dir, track_name)
            
            #Check if unlearning was applied (branches exist)
            #post-unlearning model from branches if present, else aggregated track model
            track_unlearning_dir = os.path.join(track_dir, "unlearning")
            branches_dir = os.path.join(track_unlearning_dir, "branches") if os.path.exists(track_unlearning_dir) else None
            
            model_path = None
            model_source = ""
            
            # post-unlearning model from branches
            if branches_dir and os.path.exists(branches_dir):
                # Check for comparison.json (multi-strategy mode)
                comparison_path = os.path.join(track_unlearning_dir, "comparison.json")
                if os.path.exists(comparison_path):
                    # Multi-strategy: use the best strategy's model
                    try:
                        with open(comparison_path, 'r') as f:
                            comparison = json.load(f)
                        #Get the best strategy (usually exact_retraining as baseline)
                        best_strategy = comparison.get("best_strategy", "exact_retraining")
                        branch_model_path = os.path.join(branches_dir, best_strategy, "model.pt")
                        if os.path.exists(branch_model_path):
                            model_path = branch_model_path
                            model_source = f"branch:{best_strategy}"
                            print(f"Using post-unlearning model from branch '{best_strategy}' for track {track_name}")
                    except Exception as e:
                        print(f"Warning: Could not load comparison.json for track {track_name}: {e}")
                
                #If no comparison.json, check for single strategy branch
                if not model_path:
                    branches = [d for d in os.listdir(branches_dir) if os.path.isdir(os.path.join(branches_dir, d))]
                    if branches:
                        # first available branch (usually one in single-strategy mode)
                        branch_name = branches[0]
                        branch_model_path = os.path.join(branches_dir, branch_name, "model.pt")
                        if os.path.exists(branch_model_path):
                            model_path = branch_model_path
                            model_source = f"branch:{branch_name}"
                            print(f"Using post-unlearning model from branch '{branch_name}' for track {track_name}")
            
            # Fallback: prefer aggregated track model, then global aggregated
            if not model_path:
                # try aggregated track model
                track_model_path = os.path.join(track_dir, "model.pt")
                if os.path.exists(track_model_path):
                    model_path = track_model_path
                    model_source = "track:model.pt"
                    print(f"Using aggregated track model for track {track_name}")
                else:
                    #fall back to global aggregated model
                    structure = server._get_structure_config()
                    global_model_dir = os.path.join(
                        server.results_dir,
                        structure["round_template"].format(round=round_num),
                        structure["global_model_aggregated"]
                    )
                    global_model_path = os.path.join(global_model_dir, "model.pt")
                    
                    if os.path.exists(global_model_path):
                        model_path = global_model_path
                        model_source = "global:model.pt"
                        print(f"Using global aggregated model (reference) for track {track_name}")

            if not model_path or not os.path.exists(model_path):
                print(f"Model file not found for track {track_name}")
                continue

            if not model_source:
                model_source = f"path:{os.path.basename(model_path)}"

            print(f"Evaluating track: {track_name}")

            #Load this track's model (can be a state_dict or full module)
            state_dict = None
            model_for_eval = server.global_model
            try:
                raw_obj = torch.load(model_path, map_location=server.device)
                if isinstance(raw_obj, dict):
                    state_dict = raw_obj
                elif hasattr(raw_obj, "state_dict"):
                    # Full module saved; use it directly if compatible
                    model_for_eval = raw_obj.to(server.device)
                    state_dict = raw_obj.state_dict()
                if state_dict is None:
                    raise ValueError(f"Unsupported model format for {model_path}")
                try:
                    model_for_eval.load_state_dict(state_dict)
                except Exception as e:
                    # If this looks like a SISA ensemble, rebuild a wrapper
                    if any(k.startswith("models.") for k in state_dict.keys()):
                        try:
                            model_for_eval = _build_sisa_ensemble(original_model_obj, state_dict).to(server.device)
                            model_for_eval.load_state_dict(state_dict)
                            model_source += "|sisa_ensemble_rebuilt"
                        except Exception as e2:
                            raise RuntimeError(f"Could not rebuild SISA ensemble: {e2}") from e
                    else:
                        raise
            except Exception as e:
                print(f"Warning: Strict load failed for track '{track_name}': {e}. Retrying with strict=False.")
                if state_dict is None:
                    print(f"Error: No valid state_dict available for track '{track_name}'. Skipping.")
                    continue
                try:
                    model_for_eval.load_state_dict(state_dict, strict=False)
                except Exception as e2:
                    print(f"Error: Could not load model for track '{track_name}' even with strict=False: {e2}")
                    continue
            model_for_eval.eval()

            # Initialize metrics
            test_loss = 0
            predictions = []
            actual = []
            correct = 0
            total = 0

            #Evaluate the model
            with torch.no_grad():
                for data, target in server.test_loader:
                    data, target = data.to(server.device), target.to(server.device)
                    output = model_for_eval(data)
                    loss = criterion(output, target)
                    test_loss += loss.item()

                    #For regression (N-CMAPSS)
                    if not _is_classification_task(server):
                        predictions.extend(output.cpu().numpy())
                        actual.extend(target.cpu().numpy())
                    # For classification (MNIST or tabular)
                    elif _is_classification_task(server):
                        _, predicted = torch.max(output.data, 1)
                        predictions.extend(predicted.cpu().numpy())
                        actual.extend(target.cpu().numpy())
                        total += target.size(0)
                        correct += (predicted == target).sum().item()

            # Calculate average test loss
            test_loss /= len(server.test_loader)

            # Get detailed metrics based on experiment type
            if not _is_classification_task(server):
                rmse = np.sqrt(test_loss)

                #Convert to numpy arrays for calculation
                predictions = np.array(predictions)
                actual = np.array(actual)

                #Calculate Mean Absolute Error
                mae = np.mean(np.abs(predictions - actual))

                # Calculate R²
                mean_actual = np.mean(actual)
                ss_total = np.sum((actual - mean_actual) ** 2)
                ss_residual = np.sum((actual - predictions) ** 2)
                r_squared = 1 - (ss_residual / ss_total)

                # Calculate % of predictions within 10 and 20 cycles
                within_10_cycles = np.mean(np.abs(predictions - actual) <= 10.0) * 100
                within_20_cycles = np.mean(np.abs(predictions - actual) <= 20.0) * 100

                track_results[track_name] = {
                    "rmse": rmse,
                    "mae": mae,
                    "r_squared": r_squared,
                    "within_10_cycles": within_10_cycles,
                    "within_20_cycles": within_20_cycles,
                    "model_source": model_source
                }

                print(f"Track '{track_name}' - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r_squared:.4f}")

            elif _is_classification_task(server):
                accuracy = correct / total if total > 0 else 0

                # Calculate precision, recall, and F1 score
                precision, recall, f1, _ = precision_recall_fscore_support(
                    actual, predictions, average='weighted', zero_division=0
                )

                track_results[track_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "test_loss": test_loss,
                    "model_source": model_source
                }

                print(f"Track '{track_name}' - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        #Restore the original global model state/object
        server.global_model = original_model_obj
        server.global_model.load_state_dict(original_state)

        #Save track results to a file
        if track_results and server.results_dir:
            results_path = os.path.join(
                server.output_dir,
                f"track_evaluation_round_{round_num}.json"
            )

            with open(results_path, 'w') as f:
                json.dump(make_json_serializable(track_results), f, indent=2)

            # Plot track comparisons (only in verbose mode or last round)
            is_last_round = hasattr(server, 'fl_rounds') and round_num == server.fl_rounds
            if server.verbose_plots or is_last_round:
                plot_track_comparison(server, track_results, round_num)

        return track_results

    except Exception as e:
        print(f"Error evaluating track models: {e}")
        import traceback
        traceback.print_exc()
        return {}

def plot_track_comparison(server, track_results, round_num):
    """Plot a bar comparison of the tracks' performance for this round."""
    if not track_results:
        return

    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(server.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # special case: only global track (expired disagreements)
    is_global_only = len(track_results) == 1 and "global" in track_results

    #load previous round to check if disagreements expired
    prev_results = None
    if is_global_only and round_num > 1:
        #Try to load previous round results
        prev_results_path = os.path.join(
            server.output_dir,
            f"track_evaluation_round_{round_num-1}.json"
        )
        if os.path.exists(prev_results_path):
            try:
                with open(prev_results_path, 'r') as f:
                    prev_results = json.load(f)
            except Exception as e:
                print(f"Failed to load previous round results: {e}")

    # Plot based on experiment type
    if not _is_classification_task(server):
        # RUL prediction metrics comparison
        metrics = ['rmse', 'r_squared', 'within_10_cycles', 'within_20_cycles']
        titles = ['RMSE (lower is better)', 'R² (higher is better)',
                 'Within ±10 cycles %', 'Within ±20 cycles %']

        plt.figure(figsize=(12, 10))

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            plt.subplot(2, 2, i+1)

            # Extract metric values for each track
            track_names = list(track_results.keys())
            metric_values = [track_results[track]['rmse'] if metric == 'rmse' else track_results[track][metric]
                             for track in track_names]

            #Create bar chart
            bars = plt.bar(track_names, metric_values)

            #Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

            plt.title(title)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        plt.suptitle(f'Track Performance Comparison - Round {round_num}', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'track_comparison_rul_round_{round_num}.png'),
                    bbox_inches='tight')
        plt.close()

    elif server.experiment_type == "mnist":
        # Classification metrics comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        # expired disagreements with prior results:
        #Create a special plot showing the transition
        if is_global_only and prev_results and len(prev_results) > 1:
            plt.figure(figsize=(12, 10))

            #Get last round's tracks and current global track
            tracks_to_compare = list(prev_results.keys())

            if "global" in tracks_to_compare:
                tracks_to_compare.remove("global")
                tracks_to_compare = ["global"] + tracks_to_compare

            # Create a plot comparing previous round tracks with current global
            for i, (metric, title) in enumerate(zip(metrics, titles)):
                plt.subplot(2, 2, i+1)

                # Extract values for comparison
                previous_values = [prev_results[track][metric] for track in tracks_to_compare]
                current_value = track_results["global"][metric]

                # Create a new list with previous tracks and current global
                all_tracks = tracks_to_compare + ["global (current)"]
                all_values = previous_values + [current_value]

                #Create bar chart
                bars = plt.bar(all_tracks, all_values)

                #Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate(f'{height:.4f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

                plt.title(title)
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1.1)  # Scale for classification metrics
                plt.grid(axis='y')
                plt.tight_layout()

            plt.suptitle(f'Track Performance Comparison - Round {round_num} (Disagreements Expired)', y=1.02, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'track_comparison_mnist_round_{round_num}.png'),
                       bbox_inches='tight')
            plt.close()

        # Standard comparison plot
        else:
            plt.figure(figsize=(12, 10))

            for i, (metric, title) in enumerate(zip(metrics, titles)):
                plt.subplot(2, 2, i+1)

                #Extract metric values for each track
                track_names = list(track_results.keys())
                metric_values = [track_results[track][metric] for track in track_names]

                #Create bar chart
                bars = plt.bar(track_names, metric_values)

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.annotate(f'{height:.4f}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),  # 3 points vertical offset
                                 textcoords="offset points",
                                 ha='center', va='bottom')

                plt.title(title)
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1.1)  # Scale for classification metrics
                plt.grid(axis='y')
                plt.tight_layout()

            subtitle = "Disagreements Expired - All Clients on Global Track" if is_global_only else ""
            plt.suptitle(f'Track Performance Comparison - Round {round_num}\n{subtitle}', y=1.02, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'track_comparison_mnist_round_{round_num}.png'),
                       bbox_inches='tight')
            plt.close()

def plot_track_progress(server, round_num, should_plot_all=True):
    """Plot how each track's metric evolves across rounds."""
    if "track_results" not in server.training_history or not server.training_history["track_results"]:
        print("No track results found in training history - cannot create progress plots")
        return

    #Get track results from all rounds
    track_history = server.training_history["track_results"]

    #needs >=2 rounds of track data
    if len(track_history) < 2:
        print(f"Only {len(track_history)} rounds of track data found - need at least 2 to create progress plots")
        return

    print(f"Creating track progress plots with data from {len(track_history)} rounds")

    # Plot directory
    plots_dir = os.path.join(server.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get all rounds with track data
    rounds = sorted([int(r) for r in track_history.keys()])

    # Find all track names across all rounds
    all_tracks = set()
    for r in rounds:
        if str(r) in track_history:
            all_tracks.update(track_history[str(r)].keys())

    all_tracks = sorted(list(all_tracks))

    #Define metrics based on experiment type
    if _is_classification_task(server):
        metrics = [
            {"name": "accuracy", "title": "Accuracy"},
            {"name": "precision", "title": "Precision"},
            {"name": "recall", "title": "Recall"},
            {"name": "f1", "title": "F1 Score"}
        ]
    elif not _is_classification_task(server):
        metrics = [
            {"name": "rmse", "title": "RMSE"},
            {"name": "r_squared", "title": "R²"},
            {"name": "within_10_cycles", "title": "Within ±10 cycles %"},
            {"name": "within_20_cycles", "title": "Within ±20 cycles %"}
        ]
    else:
        print(f"Unknown experiment type: {server.experiment_type}")
        return

    #Create a figure for each metric (only in verbose mode)
    if should_plot_all:
        for metric_info in metrics:
            plt.figure(figsize=(12, 6))

            metric = metric_info["name"]
            title = metric_info["title"]

            # For each track, plot its metric over time
            for track in all_tracks:
                track_values = []
                valid_rounds = []

                # Collect metric values across rounds for this track
                for r in rounds:
                    r_str = str(r)
                    if r_str in track_history and track in track_history[r_str]:
                        try:
                            # Some tracks might be missing in certain rounds
                            track_values.append(track_history[r_str][track][metric])
                            valid_rounds.append(r)
                        except KeyError:
                            continue

                #plot only if data present
                if valid_rounds and track_values:
                    plt.plot(valid_rounds, track_values, marker='o', markersize=4, label=track)

            #Add global model metric if available
            if server.experiment_type == "mnist" and metric == "accuracy" and len(server.training_history.get("global_test_accuracy", [])) > 0:
                # add Global Model line only if no 'global' track present
                if 'global' not in all_tracks:
                    # Filter only rounds that match track rounds
                    global_values = []
                    for i, r in enumerate(server.training_history["rounds"]):
                        if r in rounds:
                            global_values.append(server.training_history["global_test_accuracy"][i])

                    if global_values:
                        plt.plot(rounds[:len(global_values)], global_values, marker='s', linestyle='--',
                                 color='black', linewidth=2, label='Global Model')

            elif not _is_classification_task(server) and metric == "rmse" and len(server.training_history.get("global_test_loss", [])) > 0:
                # add Global Model line only if no 'global' track present
                if 'global' not in all_tracks:
                    #Filter only rounds that match track rounds
                    global_values = []
                    for i, r in enumerate(server.training_history["rounds"]):
                        if r in rounds:
                            global_values.append(server.training_history["global_test_loss"][i])

                    if global_values:
                        plt.plot(rounds[:len(global_values)], global_values, marker='s', linestyle='--',
                                 color='black', linewidth=2, label='Global Model')

            plt.title(f'Track {title} Over Rounds')
            plt.xlabel('Round')
            plt.ylabel(title)
            plt.grid(True)
            plt.legend(loc='best')

            #Set x-axis to show only whole numbers
            if rounds:
                plt.xticks(rounds)

            # Save figure
            plt.savefig(os.path.join(plots_dir, f'track_progress_{metric}_round_{round_num}.png'),
                       bbox_inches='tight')
            plt.close()

    # Create a multi-metric plot for comparison
    # plot only in verbose mode or last round
    is_last_round = hasattr(server, 'fl_rounds') and round_num == server.fl_rounds
    if should_plot_all or is_last_round:

        #broken axes? (track data starts at round 1)
        #visual break: round 0 was skipped
        use_broken_axes = len(rounds) > 0 and min(rounds) > 0

        if use_broken_axes:
            # Create figure with broken axes
            fig = plt.figure(figsize=(15, 10))

            # Different subplot layout based on number of metrics
            rows = 2 if len(metrics) <= 4 else 3
            cols = 2 if len(metrics) <= 4 else (3 if len(metrics) <= 9 else 4)

                                    # Define x-axis limits: narrow round 0 section, then gap, then track rounds
            min_track_round = min(rounds) if rounds else 1
            max_track_round = max(rounds) if rounds else 1

            #Create broken x-axis limits: (-0.1, 0.1) for narrow round 0
            #section, then (min_track_round-0.1, max_track_round+0.1) for
            # tracks
            xlims = ((-0.1, 0.1), (min_track_round - 0.1, max_track_round + 0.1))

            # Create a GridSpec for proper subplot management
            from matplotlib import gridspec
            gs = gridspec.GridSpec(rows, cols, figure=fig)

            for i, metric_info in enumerate(metrics[:rows*cols]):  # Limit to fit subplot grid
                metric = metric_info["name"]
                title = metric_info["title"]

                #Collect all y-values to determine y-axis limits
                all_y_values = []

                #Collect track values
                track_data = {}
                for track in all_tracks:
                    track_values = []
                    valid_rounds = []

                    for r in rounds:
                        r_str = str(r)
                        if r_str in track_history and track in track_history[r_str]:
                            try:
                                value = track_history[r_str][track][metric]
                                track_values.append(value)
                                valid_rounds.append(r)
                                all_y_values.append(value)
                            except KeyError:
                                continue

                    if valid_rounds and track_values:
                        track_data[track] = (valid_rounds, track_values)

                # Determine y-axis limits with some padding
                if all_y_values:
                    y_min = min(all_y_values)
                    y_max = max(all_y_values)
                    y_padding = (y_max - y_min) * 0.1
                    ylims = (y_min - y_padding, y_max + y_padding)
                else:
                    ylims = (0, 1)

                # Calculate subplot position
                row = i // cols
                col = i % cols

                # Create broken axes subplot (if available, otherwise use regular subplot)
                if brokenaxes is not None:
                    bax = brokenaxes(
                        xlims=xlims,
                        ylims=(ylims,),  #Single y-axis range
                        hspace=0.05,
                    subplot_spec=gs[row, col],
                    fig=fig,
                    despine=False,
                    diag_color='none'  #Remove the diagonal slashes from the broken axes (they are buggy; positioned in random places)
                )

                # Plot track data
                for track, (valid_rounds, track_values) in track_data.items():
                    bax.plot(valid_rounds, track_values, marker='o', markersize=4, label=track)

                bax.set_title(title, fontsize=21)
                bax.set_xlabel('Round', fontsize=18, labelpad=25)

                # Set different y-label padding based on metric and experiment type
                if not _is_classification_task(server):
                    if metric == "rmse":
                        ylabel_pad = 30
                    elif metric == "within_10_cycles":
                        ylabel_pad = 30
                    elif metric == "within_20_cycles":
                        ylabel_pad = 30
                    else:  # r_squared
                        ylabel_pad = 45
                else:  #mnist
                    ylabel_pad = 50

                bax.set_ylabel(title, fontsize=18, labelpad=ylabel_pad)
                bax.grid(True)

                #Set tick font sizes and customize x-axis ticks
                if brokenaxes is not None and hasattr(bax, 'axs'):
                    for ax_idx, ax in enumerate(bax.axs):
                        ax.tick_params(axis='both', which='major', labelsize=16)
                else:
                    # Regular subplot (fallback when brokenaxes not available)
                    bax.tick_params(axis='both', which='major', labelsize=16)
                    # Set x-ticks to show rounds
                    if rounds:
                        bax.set_xticks(rounds)
                        bax.set_xticklabels([str(r) for r in rounds])
                    # Format y-axis based on experiment type and metric
                    if server.experiment_type == "mnist":
                        bax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
                    elif not _is_classification_task(server):
                        if metric in ["rmse", "within_10_cycles", "within_20_cycles"]:
                            bax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
                        elif metric == "r_squared":
                            bax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

                #Only add legend to the first subplot to save space
                if i == 0:
                    if server.experiment_type == "mnist":
                        bax.legend(loc='lower right', fontsize=14)
                    elif not _is_classification_task(server):
                        bax.legend(loc='upper right', fontsize=14)
                    else:
                        bax.legend(loc='best', fontsize=14)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'track_metrics_comparison_round_{round_num}.png'),
                       bbox_inches='tight')
            plt.close()

        else:
            #Use regular matplotlib subplots
            plt.figure(figsize=(15, 10))

            # Different subplot layout based on number of metrics
            rows = 2 if len(metrics) <= 4 else 3
            cols = 2 if len(metrics) <= 4 else (3 if len(metrics) <= 9 else 4)

            for i, metric_info in enumerate(metrics[:rows*cols]):  # Limit to fit subplot grid
                plt.subplot(rows, cols, i+1)

                metric = metric_info["name"]
                title = metric_info["title"]

                # For each track, plot its metric over time
                for track in all_tracks:
                    track_values = []
                    valid_rounds = []

                    #Collect metric values across rounds for this track
                    for r in rounds:
                        r_str = str(r)
                        if r_str in track_history and track in track_history[r_str]:
                            try:
                                track_values.append(track_history[r_str][track][metric])
                                valid_rounds.append(r)
                            except KeyError:
                                continue

                    #plot only if data present
                    if valid_rounds and track_values:
                        plt.plot(valid_rounds, track_values, marker='o', markersize=4, label=track)

                plt.title(title, fontsize=21)
                plt.xlabel('Round', fontsize=18)
                plt.ylabel(title, fontsize=18)
                plt.grid(True)

                # Set x-axis to show only whole numbers
                if rounds:
                    plt.xticks(rounds, fontsize=16)

                # Set y-axis tick font size
                plt.yticks(fontsize=16)

                # Only add legend to the first subplot to save space
                if i == 0:
                    if server.experiment_type == "mnist":
                        plt.legend(loc='lower right', fontsize=14)
                    elif not _is_classification_task(server):
                        plt.legend(loc='upper right', fontsize=14)
                    else:
                        plt.legend(loc='best', fontsize=14)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'track_metrics_comparison_round_{round_num}.png'),
                       bbox_inches='tight')
            plt.close()

    plot_count_individual = len(metrics) if should_plot_all else 0
    plot_count_comparison = 1 if (should_plot_all or is_last_round) else 0
    total_plot_count = plot_count_individual + plot_count_comparison
    print(f"Saved {total_plot_count} track progress plot{'s' if total_plot_count != 1 else ''} for round {round_num}")

def plot_timing_metrics(server, round_num):
    """Plot the per-round timing breakdown (disagreement resolution, aggregation, ...)."""
    #timing metrics present?
    if not hasattr(server, 'aggregation_timing_history') or not server.aggregation_timing_history:
        print("No timing metrics found - cannot create timing plots")
        return

    #needs >=2 rounds of timing data
    if len(server.aggregation_timing_history) < 2:
        print(f"Only {len(server.aggregation_timing_history)} rounds of timing data found - need at least 2 to create timing plots")
        return

    print(f"Creating timing plots with data from {len(server.aggregation_timing_history)} rounds")

    # Plot directory
    plots_dir = os.path.join(server.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Extract timing data
    rounds = [entry["round"] for entry in server.aggregation_timing_history]
    has_disagreements = [entry["has_disagreements"] for entry in server.aggregation_timing_history]
    # Not used for now
    num_clients = [entry["num_clients"] for entry in server.aggregation_timing_history]

    #convert resolution time to ms so it reads nicer
    resolution_times_ms = [entry["resolution_time_seconds"] * 1000 for entry in server.aggregation_timing_history]
    aggregation_times = [entry["aggregation_time_seconds"] for entry in server.aggregation_timing_history]
    total_times = [entry["total_aggregation_time_seconds"] for entry in server.aggregation_timing_history]
    #Not used for now
    disagreement_loading_times = [entry["disagreement_loading_time_seconds"] for entry in server.aggregation_timing_history]
    track_saving_times = [entry["track_saving_time_seconds"] for entry in server.aggregation_timing_history]

    # Create timing plot with 2x3 layout
    plt.figure(figsize=(18, 10))

    # Plot 1: Total Aggregation Time
    plt.subplot(2, 3, 1)
    colors = ['red' if has_disag else 'blue' for has_disag in has_disagreements]
    bars = plt.bar(rounds, total_times, color=colors, alpha=0.7)
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.title('Total Aggregation Time')
    plt.grid(True, axis='y')

    # Add legend
    red_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='With Disagreements')
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.7, label='No Disagreements')
    plt.legend(handles=[red_patch, blue_patch])

    #Add value labels on bars
    for bar, time_val in zip(bars, total_times):
        height = bar.get_height()
        plt.annotate(f'{time_val:.3f}s',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

    #Plot 2: Disagreement Resolution Time (only for rounds with disagreements) - in milliseconds
    plt.subplot(2, 3, 2)
    disag_rounds = [r for r, has_disag in zip(rounds, has_disagreements) if has_disag]
    disag_resolution_times_ms = [t for t, has_disag in zip(resolution_times_ms, has_disagreements) if has_disag]

    if disag_rounds:
        bars = plt.bar(disag_rounds, disag_resolution_times_ms, color='orange', alpha=0.7)
        plt.xlabel('Round')
        plt.ylabel('Time (milliseconds)')
        plt.title('Disagreement Resolution Time')
        plt.grid(True, axis='y')

        # Add value labels on bars
        for bar, time_val in zip(bars, disag_resolution_times_ms):
            height = bar.get_height()
            plt.annotate(f'{time_val:.3f}ms',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    else:
        plt.text(0.5, 0.5, 'No rounds with\ndisagreements',
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Disagreement Resolution Time')

    # Plot 3: Model Aggregation Time
    plt.subplot(2, 3, 3)
    colors = ['red' if has_disag else 'blue' for has_disag in has_disagreements]
    bars = plt.bar(rounds, aggregation_times, color=colors, alpha=0.7)
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')
    plt.title('Model Aggregation Time')
    plt.grid(True, axis='y')

    # Add legend
    red_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7, label='With Disagreements')
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.7, label='No Disagreements')
    plt.legend(handles=[red_patch, blue_patch])

    #Add value labels on bars
    for bar, time_val in zip(bars, aggregation_times):
        height = bar.get_height()
        plt.annotate(f'{time_val:.3f}s',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=8)

    #Plot 4: Time series trends
    plt.subplot(2, 3, 4)
    ax1 = plt.gca()
    line1 = ax1.plot(rounds, aggregation_times, 's-', label='Aggregation Time (s)',
                     linewidth=2, markersize=4, color='blue')

    # Only plot resolution times for rounds with disagreements - in milliseconds on secondary y-axis
    lines = line1
    if disag_rounds:
        ax2 = ax1.twinx()
        line2 = ax2.plot(disag_rounds, disag_resolution_times_ms, '^-', label='Resolution Time (ms)',
                        linewidth=2, markersize=4, color='orange')
        ax2.set_ylabel('Resolution Time (ms)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        lines = line1 + line2

    ax1.set_xlabel('Round')
    ax1.set_ylabel('Aggregation Time (seconds)')
    ax1.set_title('Timing Trends Across Rounds')
    ax1.grid(True)

    # Add legend for both lines
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    # Plot 5: Resolution time as % of total time
    plt.subplot(2, 3, 5)
    disag_rounds_pct = [r for r, has_disag in zip(rounds, has_disagreements) if has_disag]
    resolution_percentages = []
    for i, (has_disag, res_time_ms, total_time) in enumerate(zip(has_disagreements, resolution_times_ms, total_times)):
        if has_disag and total_time > 0:
            #Convert resolution time back to seconds for percentage calculation
            res_time_s = res_time_ms / 1000
            percentage = (res_time_s / total_time) * 100
            resolution_percentages.append(percentage)

    if disag_rounds_pct and resolution_percentages:
        bars = plt.bar(disag_rounds_pct, resolution_percentages, color='coral', alpha=0.7)
        plt.xlabel('Round')
        plt.ylabel('Resolution Time (%)')
        plt.title('Resolution Time as % of Total Time')
        plt.grid(True, axis='y')

        #Add value labels on bars
        for bar, pct in zip(bars, resolution_percentages):
            height = bar.get_height()
            plt.annotate(f'{pct:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8)
    else:
        plt.text(0.5, 0.5, 'No rounds with\ndisagreements',
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Resolution Time as % of Total Time')

    # Plot 6: Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis('off')

    # Calculate summary stats
    avg_total_time = np.mean(total_times)
    avg_aggregation_time = np.mean(aggregation_times)
    with_disagreements_times = [t for t, has_disag in zip(total_times, has_disagreements) if has_disag]
    without_disagreements_times = [t for t, has_disag in zip(total_times, has_disagreements) if not has_disag]
    avg_with_disag = np.mean(with_disagreements_times) if with_disagreements_times else 0
    avg_without_disag = np.mean(without_disagreements_times) if without_disagreements_times else 0
    avg_resolution_time_ms = np.mean([t for t, has_disag in zip(resolution_times_ms, has_disagreements) if has_disag]) if any(has_disagreements) else 0
    avg_resolution_pct = np.mean(resolution_percentages) if resolution_percentages else 0

    summary_text = f"""
    TIMING SUMMARY:

    Total Rounds: {len(rounds)}
    Rounds with Disagreements: {sum(has_disagreements)}

    Average Total Time: {avg_total_time:.3f}s
    Average Aggregation Time: {avg_aggregation_time:.3f}s

    With Disagreements: {avg_with_disag:.3f}s
    Without Disagreements: {avg_without_disag:.3f}s

    Average Resolution Time: {avg_resolution_time_ms:.3f}ms
    Avg Resolution as % of Total: {avg_resolution_pct:.1f}%

    Overhead from Disagreements:
    {((avg_with_disag - avg_without_disag) / avg_without_disag * 100):.1f}%
    """ if avg_without_disag > 0 else f"""
    TIMING SUMMARY:

    Total Rounds: {len(rounds)}
    Rounds with Disagreements: {sum(has_disagreements)}

    Average Total Time: {avg_total_time:.3f}s
    Average Aggregation Time: {avg_aggregation_time:.3f}s
    Average Resolution Time: {avg_resolution_time_ms:.3f}ms
    Avg Resolution as % of Total: {avg_resolution_pct:.1f}%
    """

    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'disagreement_resolution_timing_round_{round_num}.png'),
               bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Saved timing plots for round {round_num}")
    print(f"Summary: Avg total time: {avg_total_time:.3f}s, Avg aggregation time: {avg_aggregation_time:.3f}s, Avg resolution time: {avg_resolution_time_ms:.3f}ms")
