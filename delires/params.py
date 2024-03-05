import os
from typing import Literal

import delires



MODE = Literal["deblur", "sisr"]

MODELS_PATH = os.path.join(delires.__path__[0], "models")
KERNELS_PATH = os.path.join(delires.__path__[0], "kernels")
CLEAN_DATA_PATH = os.path.join(delires.__path__[0], "data/clean_images")
DEGRADED_DATA_PATH = os.path.join(delires.__path__[0], "data/degraded_datasets")
RESTORED_DATA_PATH = os.path.join(delires.__path__[0], "results")
