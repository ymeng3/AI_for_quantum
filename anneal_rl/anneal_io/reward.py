# io/reward.py
import math
from common import Obs, Action

def order_score(obs: Obs) -> float:
    # simple proxy; replace with FFT SNR composite
    return 0.5*obs.sharpness + 0.5*min(1.0, obs.spacing_ratio)

def gated_goal_prob(obs: Obs, target_idx:int=2, sharp_thr:float=0.2):
    p = obs.recon_probs[target_idx]
    w = 1.0 if obs.sharpness >= sharp_thr else 0.0
    return w * p, w

def step_reward(obs_t: Obs, obs_tp1: Obs, action: Action,
                target_idx=2,
                alpha=0.5, beta=1.0, 
                lam_t=0.02, lam_e=0.002, lam_r=0.02):
    q_t = order_score(obs_t)
    q_tp1 = order_score(obs_tp1)
    dq = q_tp1 - q_t
    p_goal, w = gated_goal_prob(obs_tp1, target_idx)
    dt = action.dwell_min
    dT = abs(action.T_set - obs_t.T_curr)
    r = alpha*dq + beta*p_goal - lam_t*dt - lam_e*dT - lam_r*action.r_cmd
    return r

def terminal_bonus(obs_hist, K=3, target_idx=2, tau=0.8, sharp_thr=0.2, R_goal=2.0):
    if len(obs_hist) < K: return 0.0, False
    tail = obs_hist[-K:]
    ps = []
    ws = []
    for o in tail:
        p, w = gated_goal_prob(o, target_idx, sharp_thr)
        ps.append(p); ws.append(w)
    if sum(ps)/K >= tau and sum(ws)/K >= 0.7:
        return R_goal, True
    return 0.0, False
