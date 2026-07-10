"""CF-ST-MAPPO for Jaipur's four-intersection SUMO network.
Configured for 1-second micro-step resolution (86,400 steps/day).
"""
from __future__ import annotations
import csv, os, re, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

NODES = ("Rambagh", "Narayan_Singh_Circle", "Trimurti_Circle", "Birla_Mandir")
TLS = ("J10", "J17", "J23", "J7")
VEHICLE_COLUMNS = tuple(f"{d}_{v}" for d in ("North", "South", "East", "West")
                        for v in ("Cars", "2W", "3W", "Buses"))
CLASS_WEIGHTS = np.array([1.0, .45, .7, 2.5], dtype=np.float32)

@dataclass
class Config:
    history: int = 5
    hidden: int = 48
    actions: int = 2
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    learning_rate: float = 3e-4
    epochs: int = 3
    rollout_steps: int = 1200  # 1200 steps per trajectory chunk to save RAM
    yellow_seconds: int = 4
    min_green_seconds: int = 10
    seed: int = 7

def adjacency_from_network(net_file: str | Path) -> torch.Tensor:
    text = Path(net_file).read_text(encoding="utf-8")
    edges = re.findall(r'<edge id="([^"]+)" from="([^"]+)" to="([^"]+)"', text)
    a = np.eye(len(TLS), dtype=np.float32)
    index = {x:i for i,x in enumerate(TLS)}
    for _, source, target in edges:
        if source in index and target in index:
            a[index[source], index[target]] += 1.0
            a[index[target], index[source]] += 1.0
    degree = a.sum(1)
    return torch.tensor(a / np.sqrt(np.outer(degree, degree)), dtype=torch.float32)

class GraphGRU(nn.Module):
    def __init__(self, feature_dim: int, hidden: int):
        super().__init__()
        self.gcn = nn.Linear(feature_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
    def forward(self, history: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        spatial = torch.relu(torch.einsum("ij,btjf->btif", adj, history))
        spatial = torch.relu(self.gcn(spatial))
        b,t,n,h = spatial.shape
        _, out = self.gru(spatial.permute(0,2,1,3).reshape(b*n,t,h))
        return out[-1].reshape(b,n,h)

class TSFormer(nn.Module):
    def __init__(self, context_dim: int, hidden: int):
        super().__init__()
        self.embed = nn.Linear(context_dim, hidden)
        layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden*2,
                                           batch_first=True, dropout=.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
    def forward(self, context_history: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.embed(context_history))[:, -1]

class LocalActor(nn.Module):
    def __init__(self, state_dim: int, actions: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 96), nn.ReLU(), nn.Linear(96, 48),
                                 nn.ReLU(), nn.Linear(48, actions))
    def forward(self, state): return self.net(state)

class CentralCritic(nn.Module):
    def __init__(self, state_dim: int, nodes: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim*nodes, 256), nn.ReLU(), nn.Linear(256, 128),
                                 nn.ReLU(), nn.Linear(128, 1))
    def forward(self, states): return self.net(states.flatten(1)).squeeze(-1)

class CFSTMAPPONet(nn.Module):
    def __init__(self, traffic_dim=17, context_dim=6, config=Config()):
        super().__init__(); self.config = config
        self.spatial = GraphGRU(traffic_dim, config.hidden)
        self.context = TSFormer(context_dim, config.hidden)
        self.actor = LocalActor(config.hidden*2, config.actions)
        self.critic = CentralCritic(config.hidden*2)
    def fuse(self, traffic_history, context_history, adj):
        spatial = self.spatial(traffic_history, adj)
        context = self.context(context_history).unsqueeze(1).expand(-1, spatial.size(1), -1)
        return torch.cat((spatial, context), dim=-1)

class CSVTrafficEnv:
    def __init__(self, csv_file, config=Config()):
        self.config=config; self.rows=list(csv.DictReader(open(csv_file, encoding="utf-8")))
        self.by_time={};
        for row in self.rows: self.by_time.setdefault(row['Time'], []).append(row)
        self.times=list(self.by_time); self.reset()
        
    def _features(self, rows):
        vals=[]
        for node in NODES:
            row=next(x for x in rows if x['Node_ID']==node)
            counts=np.array([float(row[c]) for c in VEHICLE_COLUMNS], np.float32).reshape(4,4)
            pressure=(counts*CLASS_WEIGHTS).sum(axis=1)
            vals.append(np.r_[counts.flatten()/50, float(row['Wait_Time_Sec'])/120])
        return np.asarray(vals, np.float32)
        
    def _context(self):
        minute=(self.t//60)%1440; angle=2*np.pi*minute/1440
        rain=10.0 if ((self.t//60)//180)%7==0 else 0.0; visibility=1200. if rain else 5000.
        return np.array([rain/50, visibility/10000, 30/50, np.sin(angle), np.cos(angle), float(((self.t//60)//1440)%7>=5)], np.float32)
        
    def reset(self):
        self.t = 0 
        first = self._features(self.by_time[self.times[0]])
        self.hist = [first.copy() for _ in range(self.config.history)]
        self.ctx = [self._context() for _ in range(self.config.history)]
        self.active_phases = np.zeros(4, dtype=int)
        self.timers = np.zeros(4, dtype=int)
        return self._obs()
        
    def _obs(self): return np.asarray(self.hist),np.asarray(self.ctx)
    
    def step(self, actions):
        actions = np.asarray(actions)
        switched = np.zeros(4, dtype=bool)
        
        for i in range(4):
            if self.timers[i] > 0:
                self.timers[i] -= 1
            elif actions[i] != self.active_phases[i]:
                self.active_phases[i] = actions[i]
                self.timers[i] = self.config.min_green_seconds
                switched[i] = True

        current_min_idx = (self.t // 60) % len(self.times)
        current = self._features(self.by_time[self.times[current_min_idx]])
        weighted = current[:,:16].reshape(4,4,4).dot(CLASS_WEIGHTS)
        rain = self._context()[0]
        
        served = np.array([weighted[i, :2].sum() if self.active_phases[i]==0 else weighted[i, 2:].sum() for i in range(4)])
        queue = weighted.sum(1)-.45*served
        
        macro_reward = -(queue*(1+current[:,16]) + (4+20*rain)*switched).astype(np.float32)
        reward = macro_reward / 60.0  # Scale down by 1/60
        
        self.t += 1
        done = self.t >= 86400  # 24-hour cycle termination
        
        nxt_min_idx = (self.t // 60) % len(self.times)
        nxt = self._features(self.by_time[self.times[nxt_min_idx]])
        self.hist.pop(0); self.hist.append(nxt); self.ctx.pop(0); self.ctx.append(self._context())
        return self._obs(), reward, done, {"mean_queue":float(queue.mean())}

class MAPPO:
    def __init__(self, network, adjacency, config=Config(), device="cpu"):
        self.net=network.to(device); self.adj=adjacency.to(device); self.cfg=config; self.device=device
        self.opt=torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)
    def state(self, obs):
        traffic,context=obs
        return self.net.fuse(torch.tensor(traffic[None],device=self.device), torch.tensor(context[None],device=self.device), self.adj)
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            state=self.state(obs); logits=self.net.actor(state)[0]; dist=Categorical(logits=logits)
            action=logits.argmax(-1) if deterministic else dist.sample()
            return action.cpu().numpy(), dist.log_prob(action).cpu(), self.net.critic(state).cpu()
    def train(self, env, updates=72, log_dir="outputs/tensorboard/train"):
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(log_dir)
        log=[]; obs=env.reset()
        for update in range(updates):
            batch=[]
            for _ in range(self.cfg.rollout_steps):
                action,logp,value=self.act(obs); nxt,reward,done,info=env.step(action)
                batch.append((obs,action,logp.numpy(),value.item(),reward,done)); obs=env.reset() if done else nxt
            returns=[]; advantage=0.; next_value=0.
            for _,_,_,value,reward,done in reversed(batch):
                delta=reward.mean()+self.cfg.gamma*next_value*(1-done)-value
                advantage=delta+self.cfg.gamma*self.cfg.gae_lambda*(1-done)*advantage
                returns.append((advantage+value,advantage)); next_value=value
            returns,advantages=map(np.asarray,zip(*reversed(returns))); advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
            for _ in range(self.cfg.epochs):
                traffic=torch.tensor(np.stack([x[0][0] for x in batch]), dtype=torch.float32, device=self.device)
                context=torch.tensor(np.stack([x[0][1] for x in batch]), dtype=torch.float32, device=self.device)
                states=self.net.fuse(traffic, context, self.adj); actions=torch.tensor(np.stack([x[1] for x in batch]),device=self.device)
                old_logp=torch.tensor(np.stack([x[2] for x in batch]),device=self.device); ret=torch.tensor(returns,dtype=torch.float32,device=self.device)
                adv=torch.tensor(advantages,dtype=torch.float32,device=self.device)
                dist=Categorical(logits=self.net.actor(states)); ratio=torch.exp(dist.log_prob(actions)-old_logp)
                policy=-torch.minimum(ratio*adv[:,None], torch.clamp(ratio,1-self.cfg.clip,1+self.cfg.clip)*adv[:,None]).mean()
                value=nn.functional.mse_loss(self.net.critic(states),ret); entropy=dist.entropy().mean()
                loss=policy+.5*value-.01*entropy; self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), .5); self.opt.step()
            item={"update":update+1,"return":float(returns.mean()),"loss":float(loss.detach()),"queue":info['mean_queue']}
            log.append(item)
            writer.add_scalar("training/mean_return", item["return"], update+1)
            writer.add_scalar("training/loss", item["loss"], update+1)
            writer.add_scalar("traffic/mean_queue", item["queue"], update+1)
            # Print statement added here so you can see live progress in the cell output!
            print(f"Update {update+1}/{updates} completed | Return: {item['return']:.2f} | Loss: {item['loss']:.2f} | Queue: {item['queue']:.2f}")
        writer.close()
        return log
    def save_actor(self, path):
        torch.save({"state_dict":self.net.actor.state_dict(),"nodes":NODES,"tls":TLS,"note":"Actor only: decentralized deployment artifact"},path)