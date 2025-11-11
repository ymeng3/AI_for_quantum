# io/logger.py
import json, os, time
from common import Obs, Action, Info

class StepLogger:
    def __init__(self, out_dir="runs"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write_row(self, run_id:str, step_id:int, goal:str,
                  obs_in:Obs, action:Action, obs_out:Obs,
                  reward:float, done:bool, info:Info):
        row = {
            "run_id": run_id, "step_id": step_id, "goal": goal,
            "t_start": time.time(),
            "obs_in": obs_in.__dict__,
            "action": action.__dict__,
            "obs_out": obs_out.__dict__,
            "reward": reward, "done": done,
            "info": info.__dict__
        }
        with open(os.path.join(self.out_dir, f"{run_id}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
