# control/shield.py
from common import Action, Obs

class ActionShield:
    def __init__(self,
                 T_min=300, T_max=1050,
                 r_min=2, r_max=30,
                 dwell_min=2, dwell_max=20,
                 dT_max=200):
        self.T_min, self.T_max = T_min, T_max
        self.r_min, self.r_max = r_min, r_max
        self.dwell_min, self.dwell_max = dwell_min, dwell_max
        self.dT_max = dT_max

    def clamp(self, a: Action, s: Obs) -> (Action, bool):
        clamped = False
        # limit jump size wrt current T
        if abs(a.T_set - s.T_curr) > self.dT_max:
            a = Action(
                T_set=s.T_curr + self.dT_max * (1 if a.T_set > s.T_curr else -1),
                r_cmd=a.r_cmd,
                dwell_min=a.dwell_min
            )
            clamped = True
        T = min(max(a.T_set, self.T_min), self.T_max)
        r = min(max(a.r_cmd, self.r_min), self.r_max)
        d = min(max(a.dwell_min, self.dwell_min), self.dwell_max)
        clamped = clamped or (T != a.T_set or r != a.r_cmd or d != a.dwell_min)
        return Action(T, r, d), clamped
