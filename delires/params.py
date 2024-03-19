import os
from typing import Literal

import delires



TASK = Literal["deblur", "inpaint"]

MODELS_PATH = os.path.join(delires.__path__[0], "models")
OPERATORS_PATH = os.path.join(delires.__path__[0], "operators")
CLEAN_DATA_PATH = os.path.join(delires.__path__[0], "data/ffhq_small_test")
DEGRADED_DATA_PATH = os.path.join(delires.__path__[0], "data/degraded_datasets")
RESTORED_DATA_PATH = os.path.join(delires.__path__[0], "results")

DIFFPIR_NETWOKRS = [
    "diffusion_ffhq_10m", 
    "256x256_diffusion_uncond"
]
