
# common.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

ReconVec = List[float]

@dataclass
class Obs:
    # RHEED-derived
    recon_probs: ReconVec
    sharpness: float
    spacing_ratio: float
    embed_256: Optional[List[float]] = None
    # process
    T_curr: float = 25.0
    r_curr: float = 0.0
    dwell_elapsed: float = 0.0
    time_since_start: float = 0.0
    # summaries (history hints)
    T_peak_last: float = 25.0
    time_since_peak: float = 0.0
    time_above_900C: float = 0.0
    num_cycles: int = 0
    last_action: Optional[List[float]] = None

@dataclass
class Action:
    T_set: float
    r_cmd: float
    dwell_min: float

@dataclass
class Info:
    rheed_path: Optional[str] = None
    safety_clamped: bool = False
    raw_metrics: Optional[Dict] = None
    run_id: Optional[str] = None
    step_id: Optional[int] = None



