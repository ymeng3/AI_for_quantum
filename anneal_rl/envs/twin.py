# envs/twin.py
import uuid, math, random, os
from typing import Tuple
from common import Obs, Action, Info
from envs.base import AnnealEnv

class TwinEnv(AnnealEnv):
    def __init__(self, img_dir="runs", seed=0, target_label="sqrt13"):
        self.rng = random.Random(seed)
        self.img_dir = img_dir
        os.makedirs(img_dir, exist_ok=True)
        self.goal = target_label
        self.run_id = None
        self.step_id = 0
        self.s = None

    def reset(self, goal_label: str) -> Obs:
        self.goal = goal_label
        self.run_id = str(uuid.uuid4())
        self.step_id = 0
        # start “cold”
        self.s = Obs(recon_probs=[0.6,0.1,0.1,0.2],  # dummy vec
                     sharpness=0.05, spacing_ratio=1.0,
                     T_curr=25.0, r_curr=0.0, dwell_elapsed=0.0,
                     time_since_start=0.0, T_peak_last=25.0, time_since_peak=0.0,
                     time_above_900C=0.0, num_cycles=0, last_action=None)
        return self.s

    def _evolve_surface(self, s: Obs, a: Action) -> Obs:
        # crude thermal advance
        ramp_time = abs(a.T_set - s.T_curr) / max(a.r_cmd, 1e-6)
        seg_time = ramp_time + a.dwell_min
        T_next = a.T_set
        # summaries
        T_peak_last = max(s.T_peak_last, T_next)
        time_since_peak = 0.0 if T_next >= s.T_peak_last else s.time_since_peak + seg_time
        time_above_900 = s.time_above_900C + (seg_time if T_next >= 900 else 0.0)
        cycles = s.num_cycles + (1 if (a.T_set - s.T_curr) * (s.r_curr if s.r_curr!=0 else 1) < 0 else 0)

        # toy “order”/sharpness model: more order if you have had high-T exposure then moderate T
        order_gain = 0.0
        if s.T_peak_last >= 900 and 350 <= T_next <= 500:
            order_gain += 0.1 + 0.05 * (min(s.time_above_900C, 30)/30.0)
        order_gain += -0.01 * (a.r_cmd/30.0)  # too-fast ramps hurt
        sharp = max(0.0, min(1.0, s.sharpness + order_gain + self.rng.uniform(-0.01,0.01)))

        # fake recon prob (index 0=1x1, 1=2x1, 2=sqrt13, 3=others) – nudge sqrt13 if conditions good
        probs = s.recon_probs[:]
        target_idx = 2
        bump = max(0.0, 0.15*order_gain)
        probs[target_idx] = max(0.0, min(1.0, probs[target_idx] + bump))
        # renorm
        Z = sum(probs)
        probs = [p/Z for p in probs]

        return Obs(recon_probs=probs, sharpness=sharp, spacing_ratio=s.spacing_ratio,
                   T_curr=T_next, r_curr=a.r_cmd, dwell_elapsed=a.dwell_min,
                   time_since_start=s.time_since_start + seg_time,
                   T_peak_last=T_peak_last, time_since_peak=time_since_peak,
                   time_above_900C=time_above_900, num_cycles=cycles,
                   last_action=[a.T_set, a.dwell_min, a.r_cmd])

    def step(self, action: Action) -> Tuple[Obs, float, bool, Info]:
        self.step_id += 1
        s_next = self._evolve_surface(self.s, action)
        # no images; stub path
        info = Info(rheed_path=None, safety_clamped=False,
                    raw_metrics={"fft_snr": s_next.sharpness*5},
                    run_id=self.run_id, step_id=self.step_id)
        # reward is post-hoc; set 0 here
        done = (s_next.time_since_start >= 120)  # 2h cap
        self.s = s_next
        return s_next, 0.0, done, info

    def close(self): ...
