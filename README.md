# Traffic Congestion Benchmark: CF-ST-MAPPO vs. D3QN

This repository contains the evaluation pipeline comparing a Centralized-Federated Spatio-Temporal MAPPO model against a Dueling Double DQN baseline for traffic signal control in Jaipur.

## Architectures
- **CF-ST-MAPPO:** Leverages GCNs to aggregate spatial traffic features. 
- **D3QN:** Utilizes value/advantage streams to improve decision stability. 

## Running the Benchmark
1. Ensure dependencies are installed: `pip install -r requirements.txt`
2. Run the evaluation: `python main_evaluation.py`
3. Results will be saved to the `/data` folder.
