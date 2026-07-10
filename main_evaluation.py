import os, torch, numpy as np, pandas as pd, msgpack, matplotlib.pyplot as plt
from cf_st_mappo import Config, CFSTMAPPONet, CSVTrafficEnv, MAPPO, adjacency_from_network

class SimpleDQN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(17, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU())
        self.fc_val = torch.nn.Linear(64, 1); self.fc_adv = torch.nn.Linear(64, 4)
    def forward(self, x):
        x = self.net(x)
        return self.fc_val(x) + (self.fc_adv(x) - self.fc_adv(x).mean(dim=-1, keepdim=True))

def run_evaluation():
    device = torch.device("cpu")
    cfg = Config()
    env_m = CSVTrafficEnv('data/normal_data_v3.csv', cfg)
    env_d = CSVTrafficEnv('data/normal_data_v3.csv', cfg)
    
    # Load MAPPO
    adj = adjacency_from_network('data/Jaipur.net.xml').to(device)
    mappo_model = CFSTMAPPONet(config=cfg).to(device)
    mappo_trainer = MAPPO(mappo_model, adj, cfg)
    ckpt = torch.load('models/jaipur_cf_st_mappo_actor_1Hz.pt', map_location=device)
    mappo_trainer.net.actor.load_state_dict(ckpt['state_dict'])
    
    # Load DQN
    with open("models/DuelingDoubleDQNAgent_lr0.0001_model.pack", 'rb') as f:
        weights = msgpack.unpack(f, raw=False)['parameters']
    dqn_net = SimpleDQN().to(device)
    with torch.no_grad():
        for name, param in dqn_net.named_parameters():
            key = name.replace('net.', 'net.0.')
            w_key = key + ".weight" if "weight" in name else key + ".bias"
            if w_key in weights: param.copy_(torch.from_numpy(np.array(weights[w_key])).float())
            
    # Simulation
    q_mappo, q_dqn = [], []
    obs_m, obs_d = env_m.reset(), env_d.reset()
    for _ in range(86400):
        a_m, _, _ = mappo_trainer.act(obs_m, deterministic=True)
        obs_m, _, _, info_m = env_m.step(a_m)
        q_mappo.append(info_m['mean_queue'])
        
        a_d = [dqn_net(torch.tensor(obs_d[0][i][-1], dtype=torch.float32).view(1, -1)).argmax(-1).item() for i in range(len(obs_d[0]))]
        obs_d, _, _, info_d = env_d.step(a_d)
        q_dqn.append(info_d['mean_queue'])
        
    pd.DataFrame(q_mappo).to_csv("data/mappo_results.csv", index=False)
    pd.DataFrame(q_dqn).to_csv("data/dqn_results.csv", index=False)
    print("Benchmark complete. CSVs saved.")

if __name__ == "__main__":
    run_evaluation()