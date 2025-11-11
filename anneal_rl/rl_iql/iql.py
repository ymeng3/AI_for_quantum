import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple
from .dataset import JsonlSteps

# ---------------- Models ----------------
def mlp(in_dim, out_dim, hidden=512, layers=2):
    mods = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(layers-1):
        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
    mods += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*mods)

class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim+act_dim, 1)
    def forward(self, obs, act):  # (B,D),(B,A) -> (B,1)
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

class VNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = mlp(obs_dim, 1)
    def forward(self, obs):  # (B,D)->(B,1)
        return self.net(obs)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))  # global log-std
    def forward(self, obs):
        mu = self.mu(obs)
        std = self.logstd.exp().clamp(1e-3, 10.0)
        return mu, std
    def sample(self, obs):
        mu, std = self.forward(obs)
        a = mu + std * torch.randn_like(mu)
        return a, mu, std

# ---------------- IQL losses ----------------
def expectile_loss(resid, tau=0.7):
    # resid = target - V(o)
    w = torch.where(resid>=0, tau, 1-tau)
    return (w * resid.pow(2)).mean()

def advantage_weights(adv, beta=3.0):
    # w = exp(adv/beta), clipped for stability
    w = torch.exp(torch.clamp(adv / beta, max=20.0))
    return torch.clamp(w, 0.0, 100.0)

# ---------------- Trainer ----------------
class IQL:
    def __init__(self, obs_dim, act_dim, gamma=0.99, tau=0.7, beta=3.0, lr=1e-3, device="cpu"):
        self.device = device
        self.q1 = QNet(obs_dim, act_dim).to(device)
        self.q2 = QNet(obs_dim, act_dim).to(device)
        self.v  = VNet(obs_dim).to(device)
        self.pi = GaussianPolicy(obs_dim, act_dim).to(device)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()), lr=lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.opt_pi= torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.gamma, self.tau, self.beta = gamma, tau, beta

    @torch.no_grad()
    def _target_q(self, o2, r, done):
        v2 = self.v(o2).squeeze(-1)
        tgt = r + self.gamma * (1.0 - done.float()) * v2
        return tgt

    def update(self, batch, iters=1):
        o, a, r, o2, d = [t.to(self.device) for t in batch]
        for _ in range(iters):
            # V update (expectile regression)
            with torch.no_grad():
                q1a = self.q1(o, a).squeeze(-1)
                q2a = self.q2(o, a).squeeze(-1)
                qmin = torch.min(q1a, q2a)
            v = self.v(o).squeeze(-1)
            v_loss = expectile_loss(qmin - v, tau=self.tau)
            self.opt_v.zero_grad(); v_loss.backward(); self.opt_v.step()

            # Q update (TD)
            with torch.no_grad():
                tgt = self._target_q(o2, r, d)
            q1 = self.q1(o, a).squeeze(-1)
            q2 = self.q2(o, a).squeeze(-1)
            q_loss = F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt)
            self.opt_q.zero_grad(); q_loss.backward(); self.opt_q.step()

            # Policy update (advantage-weighted regression)
            mu, std = self.pi(o)
            # treat dataset action as target mean, weight by positive advantages
            with torch.no_grad():
                v = self.v(o).squeeze(-1)
                q1a = self.q1(o, a).squeeze(-1)
                q2a = self.q2(o, a).squeeze(-1)
                adv = torch.min(q1a, q2a) - v
                w = advantage_weights(adv, beta=self.beta)
            # L2 to dataset actions (mean squared error), weighted by w
            pi_loss = (w.unsqueeze(-1) * (mu - a).pow(2)).mean()
            self.opt_pi.zero_grad(); pi_loss.backward(); self.opt_pi.step()

        return dict(v_loss=v_loss.item(), q_loss=q_loss.item(), pi_loss=pi_loss.item())

    def act(self, obs_vec, clamp=None):
        x = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.pi(x)
            a = mu  # use mean at test-time for stability
        a = a.squeeze(0).cpu().numpy().tolist()
        # optional clamp to action bounds order: [T_set, dwell, r]
        if clamp:
            a[0] = min(max(a[0], clamp["T_min"]), clamp["T_max"])
            a[1] = min(max(a[1], clamp["dwell_min"]), clamp["dwell_max"])
            a[2] = min(max(a[2], clamp["r_min"]), clamp["r_max"])
        return a

# ------------- Training entrypoint & Policy wrapper -------------
def _collate(batch):
    import torch
    o, a, r, o2, d = zip(*batch)
    return (torch.tensor(o, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(o2, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32))

class Policy:
    """ Wraps a trained IQL policy with predict(obs)->Action. """
    def __init__(self, iql: IQL, clamp=None):
        self.iql = iql
        self.clamp = clamp or {"T_min":300,"T_max":1050,"dwell_min":2,"dwell_max":20,"r_min":2,"r_max":30}

    def predict(self, obs_dataclass) -> "Action":
        from common import Action
        # flatten obs Dataclass -> vector (use same rules as dataset)
        from .dataset import _flatten_obs
        obs_vec = _flatten_obs(obs_dataclass.__dict__)
        a = self.iql.act(obs_vec, clamp=self.clamp)
        return Action(T_set=a[0], dwell_min=a[1], r_cmd=a[2])

def train_iql(jsonl_paths, steps=200_000, batch_size=1024, device="cpu"):
    ds = JsonlSteps(jsonl_paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=_collate)
    iql = IQL(obs_dim=ds.obs_dim, act_dim=ds.act_dim, device=device)
    it = 0
    while it < steps:
        for batch in dl:
            logs = iql.update(batch, iters=1)
            it += 1
            if it % 1000 == 0:
                print(f"[{it}/{steps}] v={logs['v_loss']:.3f} q={logs['q_loss']:.3f} pi={logs['pi_loss']:.3f}")
            if it >= steps: break
    return iql
