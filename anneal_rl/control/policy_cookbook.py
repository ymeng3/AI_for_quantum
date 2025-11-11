# control/policy_cookbook.py
from dataclasses import dataclass
from typing import Literal, Optional
from common import Obs, Action

Mode = Literal["rt13", "htr"]

@dataclass
class CookbookParams:
    # shared safety-like bounds (final clamp still handled by ActionShield)
    T_high_min: float = 930.0
    T_high_max: float = 1100.0
    # --- rt13 defaults ---
    rt13_T_high: float = 980.0          # activation peak
    rt13_r_high: float = 12.0           # °C/min up to peak
    rt13_d_high: float = 20.0           # min dwell at peak
    rt13_T_target: float = 440.0        # ordering band center
    rt13_r_cool: float = 7.0            # slow cool
    rt13_d_target: float = 12.0         # stabilize at target
    rt13_band_lo: float = 400.0         # ordering window
    rt13_band_hi: float = 550.0
    rt13_min_time_in_band: float = 25.0 # ensure enough minutes spent between 550→400
    # adaptivity
    sharp_thr: float = 0.18             # gate for trusting “good progress”
    dsharp_pos: float = 0.002           # what counts as “improving”
    dsharp_neg: float = -0.001          # what counts as “getting worse”
    extend_dwell_min: float = 5.0       # min extra dwell if flat/neg trend

    # --- htr defaults ---
    htr_T_high: float = 1060.0
    htr_r_high: float = 15.0
    htr_d_high: float = 40.0
    htr_T_quench: float = 650.0
    htr_r_cool: float = 20.0
    htr_d_quench: float = 6.0


class CookbookHeuristic:
    """
    Two playbooks:
      - 'rt13': high-T activation → slow cool through 550→400 °C → hold near 440 °C.
      - 'htr' : extended high-T dwell (~1060 °C) → faster cool to ~650 °C.

    Light adaptivity:
      - extend peak dwell if sharpness still rising at peak
      - enforce minimum time in (550→400 °C) window for rt13
      - small dwell trims if trend is flat/negative at target
    """
    def __init__(self, mode: Mode = "rt13", params: Optional[CookbookParams] = None):
        assert mode in ("rt13", "htr")
        self.mode = mode
        self.P = params or CookbookParams()
        # internal phase state
        self.phase = 0
        self.time_in_order_band = 0.0
        self.prev_sharp = None

    def _dsharp(self, obs: Obs) -> float:
        if self.prev_sharp is None:
            return 0.0
        return obs.sharpness - self.prev_sharp

    def _in_rt13_band(self, T: float) -> bool:
        return self.P.rt13_band_lo <= T <= self.P.rt13_band_hi

    def predict(self, obs: Obs) -> Action:
        """
        Emits a macro action [T_set, dwell_min, r_cmd].
        Phase progression:
          rt13: (0) ramp_to_peak → (1) dwell_peak → (2) slow_cool → (3) stabilize
          htr : (0) ramp_to_peak → (1) long_dwell → (2) fast_cool → (3) short_hold
        """
        dsharp = self._dsharp(obs)

        if self.mode == "rt13":
            # --- Phase 0: ramp to activation peak ---
            if self.phase == 0:
                self.prev_sharp = obs.sharpness
                self.phase = 1
                return Action(T_set=self.P.rt13_T_high, dwell_min=max(3.0, self.P.rt13_d_high * 0.25), r_cmd=self.P.rt13_r_high)

            # --- Phase 1: dwell at peak (extend a bit if still improving) ---
            if self.phase == 1:
                dwell = self.P.rt13_d_high
                # If sharpness is clearly improving at the peak, add a small extension once
                if dsharp > self.P.dsharp_pos and obs.sharpness >= self.P.sharp_thr:
                    dwell += self.P.extend_dwell_min
                self.prev_sharp = obs.sharpness
                self.phase = 2
                return Action(T_set=self.P.rt13_T_high, dwell_min=dwell, r_cmd=self.P.rt13_r_high)

            # --- Phase 2: slow cool through ordering window, accumulate time in band ---
            if self.phase == 2:
                # Accumulate time spent in 550→400 °C band (approx by last dwell)
                if self._in_rt13_band(obs.T_curr):
                    # We don't know last dwell exactly; use a small chunk each action to enforce band time
                    self.time_in_order_band += max(0.0, obs.dwell_elapsed)
                # If we haven't spent enough time in band, keep stepping down with slow ramp
                if self.time_in_order_band < self.P.rt13_min_time_in_band or obs.T_curr > self.P.rt13_T_target + 15.0:
                    # Step down toward target with slow ramp; modest dwell per step
                    self.prev_sharp = obs.sharpness
                    return Action(T_set=self.P.rt13_T_target, dwell_min= max(6.0, self.P.rt13_d_target * 0.5), r_cmd=self.P.rt13_r_cool)
                # Enough time in band: move to stabilize
                self.prev_sharp = obs.sharpness
                self.phase = 3
                return Action(T_set=self.P.rt13_T_target, dwell_min=self.P.rt13_d_target, r_cmd=self.P.rt13_r_cool)

            # --- Phase 3: stabilize near target; small adaptive trims ---
            if self.phase == 3:
                dwell = self.P.rt13_d_target
                # If trend is slightly negative or flat and sharpness not high enough, extend dwell once
                if dsharp < self.P.dsharp_neg or (abs(dsharp) < 1e-3 and obs.sharpness < self.P.sharp_thr):
                    dwell += self.P.extend_dwell_min
                self.prev_sharp = obs.sharpness
                return Action(T_set=self.P.rt13_T_target, dwell_min=dwell, r_cmd=max(5.0, self.P.rt13_r_cool - 1.0))

        else:  # HTR mode
            # --- Phase 0: ramp to high T quickly (within allowed r_high) ---
            if self.phase == 0:
                self.prev_sharp = obs.sharpness
                self.phase = 1
                return Action(T_set=self.P.htr_T_high, dwell_min=max(5.0, self.P.htr_d_high * 0.25), r_cmd=self.P.htr_r_high)

            # --- Phase 1: long high-T dwell (reduction) ---
            if self.phase == 1:
                dwell = self.P.htr_d_high
                # If sharpness keeps rising (rare at very high T), cap extension modestly
                if dsharp > self.P.dsharp_pos:
                    dwell += min(self.P.extend_dwell_min, 5.0)
                self.prev_sharp = obs.sharpness
                self.phase = 2
                return Action(T_set=self.P.htr_T_high, dwell_min=dwell, r_cmd=self.P.htr_r_high)

            # --- Phase 2: faster cool to ~650 °C, shorter dwell ---
            if self.phase == 2:
                self.prev_sharp = obs.sharpness
                self.phase = 3
                return Action(T_set=self.P.htr_T_quench, dwell_min=self.P.htr_d_quench, r_cmd=self.P.htr_r_cool)

            # --- Phase 3: short hold; prevent drifting into 450–500 band for long ---
            if self.phase == 3:
                # keep a short hold then finish
                self.prev_sharp = obs.sharpness
                return Action(T_set=self.P.htr_T_quench, dwell_min=self.P.htr_d_quench, r_cmd=self.P.htr_r_cool)

        # Fallback (shouldn’t hit)
        self.prev_sharp = obs.sharpness
        return Action(T_set=obs.T_curr, dwell_min=5.0, r_cmd=10.0)
