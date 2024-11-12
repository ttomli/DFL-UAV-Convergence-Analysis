# src/test_aggregation.py
import torch
from model import SimpleCNN
from utils import aggregate_models

def test_aggregation():
    model1 = SimpleCNN()
    model2 = SimpleCNN()
    # Simulate training updates
    for param in model1.parameters():
        param.data.add_(torch.randn_like(param))
    for param in model2.parameters():
        param.data.add_(torch.randn_like(param))

    # Simulate batch counts
    batch_count1 = 100  # Number of batches processed by model1
    batch_count2 = 150  # Number of batches processed by model2

    aggregated_state_dict = aggregate_models(
        [model1, model2],
        batch_counts=[batch_count1, batch_count2]
    )
    aggregated_model = SimpleCNN()
    aggregated_model.load_state_dict(aggregated_state_dict)

    print("Model aggregation completed.")

if __name__ == "__main__":
    test_aggregation()