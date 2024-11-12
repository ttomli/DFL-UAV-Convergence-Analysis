# src/uav.py
import torch
import torch.nn as nn
import numpy as np
from data_loader import get_non_iid_data
import time
from collections import OrderedDict

class UAV:
    def __init__(self, uav_id, position, model, total_uavs=6):
        self.uav_id = uav_id
        self.position = np.array(position)
        self.model = model
        self.total_uavs = total_uavs
        self.train_loader, self.validation_loader = get_non_iid_data(uav_id, total_uavs)
        self.local_data_size = len(self.train_loader.dataset)
        self.neighbors = []
        self.converged = False
        self.cpu_frequency = np.random.uniform(1.0, 2.0)  # GHz, random for each UAV
        self.communication_time = 0.0
        self.computation_time = 0.0
        self.total_latency = 0.0
        self.received_models = {}  # Stores models from neighbors who have converged
        self.validation_loss = None
        self.previous_validation_loss = None
        self.validation_accuracy = None

    def move(self):
        # Simulate small random movement
        movement_vector = np.random.uniform(-5, 5, size=2)
        self.position += movement_vector
        # Keep UAVs within bounds
        self.position = np.clip(self.position, 0, 500)

    def local_training(self, epochs=1, device='cpu', mu=0.01):
        if self.converged:
            # If already converged, no need to train further
            return

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.025)
        self.model.to(device)
        self.model.train()

        # Save the initial global model parameters
        global_params = {name: param.clone().detach() for name, param in self.model.named_parameters()}

        # Simulate computation time
        start_time = time.time()

        for _ in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Add proximal term for FedProx
                prox_loss = 0.0
                for name, param in self.model.named_parameters():
                    prox_loss += ((mu / 2) * torch.norm(param - global_params[name]) ** 2)
                loss += prox_loss

                loss.backward()
                optimizer.step()

        end_time = time.time()
        # Estimate computation time inversely proportional to CPU frequency
        self.computation_time = (end_time - start_time) / self.cpu_frequency

        # After training, validate and check for convergence
        self.validate(device=device)
        self.check_convergence()

    def check_convergence(self, threshold=0.01):
        # Check if the improvement in validation loss is less than the threshold
        if self.previous_validation_loss is not None:
            improvement = self.previous_validation_loss - self.validation_loss
            if improvement < threshold:
                self.converged = True
        self.previous_validation_loss = self.validation_loss

    def communicate_with_neighbors(self, device='cpu'):
        if self.converged:
            # Broadcast model to neighbors
            for neighbor in self.neighbors:
                # Simulate communication time based on distance
                distance = np.linalg.norm(self.position - neighbor.position)
                communication_speed = 1e6  # Adjust as needed (e.g., speed in units per second)
                comm_time = distance / communication_speed
                self.communication_time += comm_time

                # Send model to neighbor
                neighbor.receive_model(self.uav_id, self.model.state_dict())

        # Aggregate models received from neighbors who have converged
        if self.received_models:
            models_state_dicts = [self.model.state_dict()]  # Include own model
            sample_counts = [self.get_sample_count()]  # Include own sample count

            for neighbor_id, state_dict in self.received_models.items():
                models_state_dicts.append(state_dict)
                sample_counts.append(1)  # Adjust weighting if needed

            aggregated_state_dict = self.aggregate_models(models_state_dicts, sample_counts)
            self.model.load_state_dict(aggregated_state_dict)

            # Clear received models after aggregation
            self.received_models = {}

        # Update total latency
        self.total_latency = self.communication_time + self.computation_time
        # Reset communication time for next round
        self.communication_time = 0.0

    def receive_model(self, neighbor_id, state_dict):
        # Store the received model if neighbor has converged
        self.received_models[neighbor_id] = state_dict

    def aggregate_models(self, models_state_dicts, sample_counts):
        aggregated_state_dict = OrderedDict()
        total_samples = sum(sample_counts)
        model_keys = models_state_dicts[0].keys()

        for key in model_keys:
            params = [state_dict[key] for state_dict in models_state_dicts]
            if params[0].dtype.is_floating_point:
                # Compute weighted average of parameters
                weighted_params = torch.stack([params[i] * sample_counts[i] for i in range(len(params))], dim=0)
                aggregated_param = torch.sum(weighted_params, dim=0) / total_samples
            else:
                # For integer buffers like num_batches_tracked, take the value from one of the models
                aggregated_param = params[0]
            aggregated_state_dict[key] = aggregated_param

        return aggregated_state_dict

    def validate(self, device='cpu'):
        self.model.to(device)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.validation_loss = val_loss / total
        self.validation_accuracy = 100.0 * correct / total

    def get_sample_count(self):
        return self.local_data_size