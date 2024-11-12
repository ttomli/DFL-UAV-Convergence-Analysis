# src/utils.py
from collections import OrderedDict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def evaluate_model(model, data_loader, device='cpu'):
    """
    Evaluates the model on the provided data loader.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (str): Device to run the evaluation on ('cpu' or 'cuda').

    Returns:
        average_loss (float): Average loss over the dataset.
        accuracy (float): Accuracy percentage.
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return average_loss, accuracy

def aggregate_models(uav_models, sample_counts):
    """
    Aggregates models by computing a weighted average of their parameters based on sample counts.

    Args:
        uav_models (list): List of nn.Module models from UAVs.
        sample_counts (list): List of the number of training samples at each UAV.

    Returns:
        aggregated_state_dict (OrderedDict): State dict of the aggregated model.
    """
    state_dicts = [model.state_dict() for model in uav_models]
    aggregated_state_dict = OrderedDict()
    total_samples = sum(sample_counts)
    model_keys = state_dicts[0].keys()

    for key in model_keys:
        params = [state_dict[key] for state_dict in state_dicts]
        if params[0].dtype.is_floating_point:
            # Compute weighted average of parameters
            weighted_params = torch.stack([params[i] * sample_counts[i] for i in range(len(params))], dim=0)
            aggregated_param = torch.sum(weighted_params, dim=0) / total_samples
        else:
            # For integer buffers like num_batches_tracked, take the value from one of the models
            aggregated_param = params[0]
        aggregated_state_dict[key] = aggregated_param

    return aggregated_state_dict

def plot_metrics(val_loss, val_accuracy, test_loss, test_accuracy, latency, prefix='decentralized'):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    rounds = range(1, len(val_loss) + 1)

    # Validation Loss
    plt.figure()
    plt.plot(rounds, val_loss, marker='o')
    plt.title(f'{prefix.capitalize()} FL - Average Validation Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.savefig(f'plots/{prefix}_validation_loss.png')
    plt.close()

    # Validation Accuracy
    plt.figure()
    plt.plot(rounds, val_accuracy, marker='o')
    plt.title(f'{prefix.capitalize()} FL - Average Validation Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.savefig(f'plots/{prefix}_validation_accuracy.png')
    plt.close()

    # Test Loss
    plt.figure()
    plt.plot(rounds, test_loss, marker='o')
    plt.title(f'{prefix.capitalize()} FL - Test Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.savefig(f'plots/{prefix}_test_loss.png')
    plt.close()

    # Test Accuracy
    plt.figure()
    plt.plot(rounds, test_accuracy, marker='o')
    plt.title(f'{prefix.capitalize()} FL - Test Accuracy')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.savefig(f'plots/{prefix}_test_accuracy.png')
    plt.close()

    # Latency
    if latency:
        plt.figure()
        plt.plot(rounds, latency, marker='o')
        plt.title(f'{prefix.capitalize()} FL - Average Total Latency per Round')
        plt.xlabel('Round')
        plt.ylabel('Latency (s)')
        plt.savefig(f'plots/{prefix}_latency.png')
        plt.close()

def plot_comparison_metrics(centralized_test_loss, centralized_test_accuracy, decentralized_test_loss, decentralized_test_accuracy):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    rounds = range(1, len(centralized_test_loss) + 1)

    # Plot Test Loss
    plt.figure()
    plt.plot(rounds, centralized_test_loss, label='Centralized FL', marker='o')
    plt.plot(rounds, decentralized_test_loss, label='Decentralized FL', marker='s')
    plt.title('Test Loss Comparison')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/test_loss_comparison.png')
    plt.close()

    # Plot Test Accuracy
    plt.figure()
    plt.plot(rounds, centralized_test_accuracy, label='Centralized FL', marker='o')
    plt.plot(rounds, decentralized_test_accuracy, label='Decentralized FL', marker='s')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('plots/test_accuracy_comparison.png')
    plt.close()