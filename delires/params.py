import os
from typing import Literal

import delires


MODELS_PATH = os.path.join(delires.__path__[0], "models")
OPERATORS_PATH = os.path.join(delires.__path__[0], "operators")
CLEAN_DATA_PATH = os.path.join(delires.__path__[0], "data/ffhq_test20")
DEGRADED_DATA_PATH = os.path.join(delires.__path__[0], "data/degraded_data")
RESTORED_DATA_PATH = os.path.join(delires.__path__[0], "results")

DIFFPIR_NETWOKRS = [
    "diffusion_ffhq_10m", 
    "256x256_diffusion_uncond"
]

HF_REPO_ID = "t0m1ab/mva-delires-data"

MATLAB_BLUR_KERNELS_FILES = [
    "Levin09.mat",
    "kernels_12.mat",
    "custom_blur_centered.mat",
    # "kernels_bicubicx234.mat",
]
