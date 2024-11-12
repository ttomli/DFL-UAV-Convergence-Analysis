# src/convergence.py
import torch

def has_converged(uav, threshold=0.001, window=3):
    """
    Determines if a UAV has converged locally.

    Args:
        uav (UAV): The UAV object.
        threshold (float): Convergence threshold for parameter changes.
        window (int): Number of past parameter states to consider.

    Returns:
        bool: True if the UAV has converged, False otherwise.
    """
    current_params = torch.cat([param.data.flatten() for param in uav.model.parameters()])
    uav.param_history.append(current_params)

    if len(uav.param_history) > window:
        uav.param_history.pop(0)
        diffs = [torch.norm(uav.param_history[i] - uav.param_history[i - 1]) for i in range(1, len(uav.param_history))]
        avg_diff = sum(diffs) / len(diffs)
        return avg_diff < threshold

    return False