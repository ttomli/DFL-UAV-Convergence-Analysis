Decentralized Federated Learning for UAV Networks

This repository contains a simulation of Decentralized Federated Learning (DFL) for Unmanned Aerial Vehicle (UAV) networks. The project is based on the paper:

“Decentralized Federated Learning for UAV Networks: Architecture, Challenges, and Opportunities” by Qu et al.

The simulation compares the performance of centralized and decentralized federated learning in UAV networks, incorporating factors such as communication latency, computation time, and network topology dynamics.

Table of Contents

	•	Introduction
	•	Features
	•	Installation
	•	Usage
	•	Running the Simulation
	•	Generating Plots
	•	Visualizing the UAV Network
	•	Creating a Network Simulation Video
	•	Project Structure
	•	Dependencies
	•	Results
	•	Contributing
	•	License
	•	References

Introduction

Federated Learning (FL) enables multiple clients to collaboratively train a global model without sharing their local data, preserving privacy. In UAV networks, decentralized FL allows UAVs to learn from each other without relying on a central server, which is crucial in environments where central coordination is infeasible.

This project simulates the decentralized FL process among UAVs, considering realistic factors such as:
	•	Dynamic network topology due to UAV movement.
	•	Communication latency based on distance and bandwidth.
	•	Computation time influenced by UAVs’ CPU frequencies.
	•	Convergence criteria for local models.
	•	Non-IID data distribution among UAVs.

Features

	•	Centralized and Decentralized FL Simulation: Compare the performance of centralized and decentralized federated learning.
	•	Dynamic UAV Network Topology: Simulate UAV movements and updating neighbor connections.
	•	Communication and Computation Latency Modeling: Incorporate realistic delays in model training and communication.
	•	Convergence-Based Communication: UAVs broadcast models to neighbors upon convergence.
	•	Visualization:
	•	Plot metrics such as validation loss, validation accuracy, test loss, test accuracy, and latency.
	•	Visualize the UAV network topology at each round, showing UAV positions, connections, and convergence status.
	•	Generate a video of the network simulation over time.
	•	Non-IID Data Distribution: Assign unique data slices to each UAV to simulate non-IID conditions.

Installation

Clone the Repository

git clone https://github.com/ttomli/DFL-UAV-Convergence-Analysis.git
cd DFL-UAV-Convergence-Analysis

Create a Virtual Environment

It’s recommended to use a virtual environment to manage dependencies.

python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

If requirements.txt is not provided, install the following packages:

pip install torch torchvision numpy matplotlib networkx

Usage

Running the Simulation

To run the simulation for both centralized and decentralized FL:

python src/main.py

This script will execute both simulations, collect metrics, and save plots and network visualizations.

Generating Plots

After running the simulation, plots will be saved in the plots directory:
	•	test_loss_comparison.png
	•	test_accuracy_comparison.png
	•	decentralized_validation_loss.png
	•	decentralized_validation_accuracy.png
	•	decentralized_test_loss.png
	•	decentralized_test_accuracy.png
	•	decentralized_latency.png

Visualizing the UAV Network

Network visualizations for each round are saved in the network_visualizations directory as images:
	•	network_round_001.png
	•	network_round_002.png
	•	…
	•	network_round_010.png

Each image shows:
	•	UAV Positions: Nodes represent UAVs at their current positions.
	•	Connections: Edges represent communication links between UAVs.
	•	Convergence Status:
	•	Green Nodes: UAVs that have converged.
	•	Red Nodes: UAVs that are still training.
	•	Legend: Indicates the meaning of node colors.

Creating a Network Simulation Video

You can create a video from the network visualizations using FFmpeg.

Install FFmpeg

If you don’t have FFmpeg installed, download it from ffmpeg.org and follow the installation instructions for your operating system.

Generate the Video

Navigate to the network_visualizations directory:

cd network_visualizations

Run the following command to create a video:

ffmpeg -framerate 1 -i network_round_%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p network_simulation.mp4

	•	Options:
	•	-framerate 1: Sets the input framerate to 1 frame per second.
	•	-i network_round_%03d.png: Specifies the input files.
	•	-c:v libx264: Uses the H.264 codec.
	•	-r 30: Sets the output framerate to 30 frames per second.
	•	-pix_fmt yuv420p: Sets the pixel format.
	•	network_simulation.mp4: Name of the output video file.

Return to the main directory:

cd ..

The video network_simulation.mp4 will be located in the network_visualizations directory.

Project Structure

DFL-UAV-Convergence-Analysis/
├── src/
│   ├── data_loader.py
│   ├── main.py
│   ├── model.py
│   ├── network.py
│   ├── uav.py
│   ├── utils.py
│   └── convergence.py
├── plots/
│   └── [Generated plots]
├── network_visualizations/
│   └── [Network visualization images]
├── README.md
├── requirements.txt
└── [Other files]

	•	src/: Contains the source code files.
	•	plots/: Directory where plots are saved.
	•	network_visualizations/: Directory where network visualization images are saved.

Dependencies

	•	Python 3.6 or higher
	•	PyTorch
	•	torchvision
	•	numpy
	•	matplotlib
	•	networkx
	•	FFmpeg (for creating the video)

Install Python packages using pip as shown in the Installation section.

Results

The simulation outputs include:
	•	Metrics Plots: Compare the performance of centralized and decentralized FL in terms of test loss and accuracy.
	•	Network Visualizations: Observe the dynamics of the UAV network topology and convergence status over time.
	•	Latency Analysis: Understand the impact of communication and computation delays on the learning process.

These results can help in analyzing the effectiveness of decentralized federated learning in UAV networks and understanding the challenges involved.

Contributing

Contributions are welcome! If you have suggestions for improvements or want to report issues, please open an issue or submit a pull request.

When contributing, please follow these guidelines:
	•	Fork the repository.
	•	Create a new branch for your feature or bug fix.
	•	Write clear commit messages.
	•	Test your changes thoroughly.
	•	Ensure code adheres to the existing style conventions.

License

This project is licensed under the MIT License.

References

	•	Qu, Y., Lan, T., Li, D., & Peng, M. “Decentralized Federated Learning for UAV Networks: Architecture, Challenges, and Opportunities”.