# src/communication.py
import numpy as np

def compute_data_rate(uav1, uav2):
    # Convert dBm to Watts
    transmission_power = 10 ** ((30 - 30) / 10)  # 30 dBm to Watts
    channel_gain = 10 ** ((-50 - 30) / 10)  # -50 dBm to Watts
    noise_power = 10 ** ((-90 - 30) / 10)  # -90 dBm to Watts
    bandwidth = 0.4e6  # 0.4 MHz

    # Compute SNR
    distance = np.linalg.norm(uav1.position - uav2.position)
    path_loss = channel_gain / (distance ** 2)  # Simplified path loss model
    snr = (transmission_power * path_loss) / noise_power

    # Compute data rate using Shannon-Hartley theorem
    data_rate = bandwidth * np.log2(1 + snr)
    return data_rate  # in bits per second

def transmission_time(uav1, uav2, data_size_bits):
    data_rate = compute_data_rate(uav1, uav2)
    return data_size_bits / data_rate  # in seconds
