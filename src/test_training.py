# src/test_training.py
import torch
from uav import UAV
from model import SimpleCNN

def test_training():
    model = SimpleCNN()
    position = [250, 250]
    uav = UAV(uav_id=0, position=position, model=model)
    print("Starting local training...")
    uav.local_training(device='cpu')
    print("Local training completed.")

    print("Starting validation...")
    uav.validate(device='cpu')
    print(f"Validation Loss: {uav.validation_loss:.4f}, Validation Accuracy: {uav.validation_accuracy:.2f}%")

if __name__ == "__main__":
    test_training()