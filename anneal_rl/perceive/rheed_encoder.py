# perceive/rheed_encoder.py
from typing import Dict
from common import Obs
class RHEEDEncoder:
    def __init__(self, model_ckpt=None):
        # load your classifier/metrics here (or no-op in twin)
        self.ready = True
    def features_from_image(self, img_path:str) -> Dict:
        # return dict with 'recon_probs','sharpness','spacing_ratio','embed_256'
        raise NotImplementedError
