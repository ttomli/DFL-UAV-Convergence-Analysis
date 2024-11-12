# src/network.py
import networkx as nx
import numpy as np
from uav import UAV
from model import SimpleCNN
import matplotlib.pyplot as plt
import os

LINK_AVAILABILITY_PROB = 0.8

class UAVNetwork:
    def __init__(self, num_uavs=6):
        self.graph = nx.Graph()
        self.num_uavs = num_uavs
        self.uavs = []
        self.positions = self.generate_positions()
        self.initialize_uavs()
    
    def generate_positions(self):
        print("Generating UAV positions...")
        center = np.array([250, 250])
        radius = 100  # Adjust radius as needed
        num_uavs = self.num_uavs
        angles = np.linspace(0, 2 * np.pi, num_uavs, endpoint=False)
        positions = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            positions.append(np.array([x, y]))
        return positions

    def initialize_uavs(self):
        for i, position in enumerate(self.positions):
            model = SimpleCNN()
            uav = UAV(uav_id=i, position=position, model=model)
            self.uavs.append(uav)
            self.graph.add_node(uav.uav_id, uav=uav)

    # In UAVNetwork.update_topology()
    def update_topology(self):
        self.graph.clear_edges()
        for uav in self.uavs:
            uav.move()  # Update UAV positions
            uav.neighbors = []  # Reset neighbors
        for i, uav1 in enumerate(self.uavs):
            for j, uav2 in enumerate(self.uavs):
                if i < j:
                    distance = np.linalg.norm(uav1.position - uav2.position)
                    if 80 <= distance <= 120:
                        if np.random.rand() < LINK_AVAILABILITY_PROB:
                            self.graph.add_edge(uav1.uav_id, uav2.uav_id)
                            uav1.neighbors.append(uav2)
                            uav2.neighbors.append(uav1)
    
    def visualize_network(self, round_num):
        if not os.path.exists('network_visualizations'):
            os.makedirs('network_visualizations')

        plt.figure(figsize=(8, 8))
        pos = {uav.uav_id: uav.position for uav in self.uavs}

        # Node colors based on convergence status
        node_colors = ['green' if uav.converged else 'red' for uav in self.uavs]

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=500, node_color=node_colors)

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos)

        # Labels
        labels = {uav.uav_id: f'UAV {uav.uav_id}' for uav in self.uavs}
        nx.draw_networkx_labels(self.graph, pos, labels)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Converged', markerfacecolor='green', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Not Converged', markerfacecolor='red', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title(f'UAV Network Topology at Round {round_num}')
        plt.axis('off')
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'network_visualizations/network_round_{round_num:03d}.png')
        plt.close()