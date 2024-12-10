# src/main.py
import torch
import numpy as np
from model import SimpleCNN
from network import UAVNetwork
from data_loader import get_test_data
from utils import evaluate_model, aggregate_models, plot_metrics, plot_comparison_metrics, plot_comparison_metrics_all
from convergence import has_converged

def simulate_centralized_fl(num_rounds=10):
    # Initialize lists to store metrics
    val_loss_history = []
    val_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []
    latency_history = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = UAVNetwork(num_uavs=6)
    server = network.uavs[0]  # Designate UAV 0 as the server
    clients = network.uavs[1:]
    test_loader = get_test_data()

    # Initialize lists to store metrics
    test_loss_history = []
    test_accuracy_history = []

    for round_num in range(num_rounds):
        total_latency = 0.0 # Track total latency for this round
        # Clients perform local training
        sample_counts = []
        for uav in clients:
            uav.local_training(device=device)
            # Collect the number of training samples
            sample_counts.append(uav.get_sample_count())
            total_latency += uav.total_latency

        # Clients send model updates to the server
        aggregated_state_dict = aggregate_models(
            [uav.model for uav in clients],
            sample_counts=sample_counts
        )

        # Server updates its model
        server.model.load_state_dict(aggregated_state_dict)

        # Server sends the updated model back to clients
        for uav in clients:
            uav.model.load_state_dict(server.model.state_dict())
            uav.total_latency = 0.0  # Reset latency for next round

        # Evaluate on validation sets of clients
        val_losses = []
        val_accuracies = []
        for uav in clients:
            uav.validate(device=device)
            if uav.validation_loss is not None:
                val_losses.append(uav.validation_loss)
            if uav.validation_accuracy is not None:
                val_accuracies.append(uav.validation_accuracy)

        avg_val_loss = np.mean(val_losses)
        avg_val_accuracy = np.mean(val_accuracies)
        avg_latency = total_latency / len(clients)

        # Evaluate the global model on the test dataset
        test_loss, test_accuracy = evaluate_model(server.model, test_loader, device)
        print(f"Round {round_num + 1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Store metrics
        val_loss_history.append(avg_val_loss)
        val_accuracy_history.append(avg_val_accuracy)
        latency_history.append(avg_latency)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
    return val_loss_history, val_accuracy_history, test_loss_history, test_accuracy_history, latency_history

def simulate_decentralized_fl(num_rounds=10):
    # Initialize lists to store metrics
    val_loss_history = []
    val_accuracy_history = []
    test_loss_history = []
    test_accuracy_history = []
    latency_history = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = UAVNetwork(num_uavs=6)
    test_loader = get_test_data()

    for round_num in range(num_rounds):
        network.update_topology()

        # Each UAV performs local training if not converged
        for uav in network.uavs:
            uav.local_training(device=device)

        # Each UAV communicates with neighbors
        for uav in network.uavs:
            uav.communicate_with_neighbors(device=device)

        # Collect validation metrics
        val_losses = [uav.validation_loss for uav in network.uavs if uav.validation_loss is not None]
        val_accuracies = [uav.validation_accuracy for uav in network.uavs if uav.validation_accuracy is not None]
        avg_val_loss = np.mean(val_losses)
        avg_val_accuracy = np.mean(val_accuracies)
        avg_total_latency = np.mean([uav.total_latency for uav in network.uavs])
        print(f"Round {round_num + 1}: Avg Validation Loss: {avg_val_loss:.4f}, Avg Validation Accuracy: {avg_val_accuracy:.2f}%, Avg Latency: {avg_total_latency:.4f}s")

        # Aggregate models from all UAVs for testing purposes
        aggregated_state_dict = aggregate_models(
            [uav.model for uav in network.uavs],
            sample_counts=[uav.get_sample_count() for uav in network.uavs]
        )
        aggregated_model = SimpleCNN()
        aggregated_model.load_state_dict(aggregated_state_dict)

        # Evaluate aggregated model on the test dataset
        test_loss, test_accuracy = evaluate_model(aggregated_model, test_loader, device)
        print(f"Round {round_num + 1}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Visualize network
        network.visualize_network(round_num + 1)

        # Store metrics
        val_loss_history.append(avg_val_loss)
        val_accuracy_history.append(avg_val_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        latency_history.append(avg_total_latency)
    
    # After simulation, plot metrics
    plot_metrics(val_loss_history, val_accuracy_history, test_loss_history, test_accuracy_history, latency_history, prefix='decentralized')

    return val_loss_history, val_accuracy_history, test_loss_history, test_accuracy_history, latency_history

if __name__ == "__main__":
    num_rounds = 10
    print("Simulating Centralized Federated Learning...")
    centralized_val_loss_history, centralized_val_accuracy_history, centralized_test_loss_history, centralized_test_accuracy_history, centralized_latency_history = simulate_centralized_fl(num_rounds)

    print("\nSimulating Decentralized Federated Learning...")
    decentralized_val_loss_history, decentralized_val_accuracy_history, decentralized_test_loss_history, decentralized_test_accuracy_history, decentralized_latency_history = simulate_decentralized_fl(num_rounds)

    # Compare and plot metrics side by side on the same graphs
    plot_comparison_metrics_all(
        centralized_val_loss_history, centralized_val_accuracy_history,
        centralized_test_loss_history, centralized_test_accuracy_history, centralized_latency_history,
        decentralized_val_loss_history, decentralized_val_accuracy_history,
        decentralized_test_loss_history, decentralized_test_accuracy_history, decentralized_latency_history
    )