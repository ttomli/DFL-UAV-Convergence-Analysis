# src/test_network.py
from network import UAVNetwork

def test_network():
    network = UAVNetwork(num_uavs=6)
    print("Initial UAV positions:")
    for uav in network.uavs:
        print(f"UAV {uav.uav_id}: {uav.position}")

    network.update_topology()
    print("\nUAV neighbors after topology update:")
    for uav in network.uavs:
        neighbor_ids = [neighbor.uav_id for neighbor in uav.neighbors]
        print(f"UAV {uav.uav_id} neighbors: {neighbor_ids}")

if __name__ == "__main__":
    test_network()