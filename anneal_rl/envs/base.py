# envs/base.py
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
from common import Obs, Action, Info

class AnnealEnv(ABC):
    @abstractmethod
    def reset(self, goal_label: str) -> Obs: ...
    @abstractmethod
    def step(self, action: Action) -> Tuple[Obs, float, bool, Info]: ...
    @abstractmethod
    def close(self): ...
