import json, math
from typing import List, Dict
from torch.utils.data import Dataset

def _flatten_obs(o: Dict) -> List[float]:
    # Same features as your state spec: recon_probs, sharpness, spacing_ratio, embed_256(optional),
    # process telemetry, summaries, last_action.
    x = []
    x += o["recon_probs"]
    x += [o.get("sharpness",0.0), o.get("spacing_ratio",1.0)]
    emb = o.get("embed_256", [])
    x += emb if emb else []
    x += [o.get("T_curr",0.0), o.get("r_curr",0.0), o.get("dwell_elapsed",0.0), o.get("time_since_start",0.0)]
    x += [o.get("T_peak_last",0.0), o.get("time_since_peak",0.0), o.get("time_above_900C",0.0)]
    x += [float(o.get("num_cycles",0))]
    la = o.get("last_action", [0.0,0.0,0.0])
    x += la
    return x

class JsonlSteps(Dataset):
    def __init__(self, jsonl_paths: List[str]):
        self.rows = []
        for p in jsonl_paths:
            with open(p) as f:
                for line in f:
                    row = json.loads(line)
                    if "obs_in" in row and "obs_out" in row and "action" in row:
                        self.rows.append(row)
        # infer dims
        sample_x = _flatten_obs(self.rows[0]["obs_in"])
        self.obs_dim = len(sample_x)
        self.act_dim = 3  # [T_set, dwell_min, r_cmd]

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        o  = _flatten_obs(r["obs_in"])
        a  = [r["action"]["T_set"], r["action"]["dwell_min"], r["action"]["r_cmd"]]
        rwd= float(r.get("reward",0.0))
        o2 = _flatten_obs(r["obs_out"])
        done = bool(r.get("done", False))
        return o, a, rwd, o2, done
