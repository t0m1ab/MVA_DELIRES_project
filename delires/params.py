import os
from typing import Literal

import delires



MODELS_PATH = os.path.join(delires.__path__[0], "models")
KERNELDIR = os.path.join(delires.__path__[0], "kernels")
CLEAN_DATA_PATH = os.path.join(delires.__path__[0], "data/clean_images")
DEGRAGDED_DATA_PATH = os.path.join(delires.__path__[0], "data/degraded_datasets")
RESTORED_DATA_PATH = os.path.join(delires.__path__[0], "results")