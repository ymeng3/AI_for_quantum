# control/policy_random.py
import random
from common import Obs, Action

class RandomHeuristic:
    def __init__(self, seed=0, bounds=(300,1050, 2,30, 2,20)):
        self.rng = random.Random(seed)
        self.Tmin, self.Tmax, self.rmin, self.rmax, self.dmin, self.dmax = bounds
    def predict(self, s: Obs) -> Action:
        return Action(
            T_set=self.rng.uniform(self.Tmin, self.Tmax),
            r_cmd=self.rng.uniform(self.rmin, self.rmax),
            dwell_min=self.rng.uniform(self.dmin, self.dmax),
        )

class StaircaseHeuristic:
    def __init__(self, phases=None):
        # [(T_set, dwell, r)]
        self.phases = phases or [(1000, 15, 15), (480, 5, 8), (444, 10, 5)]
        self.i = 0
    def predict(self, s: Obs) -> Action:
        a = self.phases[self.i]
        self.i = min(self.i+1, len(self.phases)-1)
        return Action(T_set=a[0], dwell_min=a[1], r_cmd=a[2])
