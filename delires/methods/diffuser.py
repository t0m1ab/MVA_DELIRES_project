import os
import numpy as np
from pathlib import Path
from logging import Logger, getLogger
from typing import Union


from delires.data import load_downsample_kernel, load_blur_kernel
from delires.utils.utils_logger import logger_info

from delires.params import (
    RESTORED_DATA_PATH,
)



class Diffuser():
    def __init__(self, logger: Logger = None, autolog: str = None, device = "cpu"):
        self.device = device
        self.logger = logger

        if autolog is not None and self.logger is None: # create a logger if not provided but if autolog is specified
            Path(RESTORED_DATA_PATH).mkdir(parents=True, exist_ok=True)
            logger_info(autolog, log_path=os.path.join(RESTORED_DATA_PATH, f"{autolog}.log"))
            self.logger = getLogger(autolog)

    def load_downsample_kernel(
        self,
        k_index: int = 0,
        cwd: str = "",
        ):
        self.kernel = load_downsample_kernel(self.classical_degradation, self.sf, k_index, cwd)
        
    def load_blur_kernel(
        self,
        diy_kernel_path: str|None = None,
        cwd: str = ""
        ):
        self.kernel, self.kernel_4d = load_blur_kernel(diy_kernel_path, self.device, cwd)
    
    def load_model(self, model_path: str):
        raise NotImplementedError

    def save_restored_image(self, path: str):
        pass
        
    def apply_debluring(self, degraded_image: np.ndarray):
        pass
    
    def apply_sisr(self, degraded_image: np.ndarray):
        pass