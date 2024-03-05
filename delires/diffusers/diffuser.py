import os
import hdf5storage
import numpy as np
import torch

from delires.data import load_downsample_kernel, load_blur_kernel


class Diffuser():
    def __init__(self, device = "cpu", **kwargs):
        self.classical_degradation = kwargs.get("sisr_classical_degradation", False)
        self.sf = kwargs.get("sf", 4)
        self.device = device

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
    
    def save_restored(self, path: str):
        pass
        
    def apply_debluring(self, degraded_image: np.ndarray):
        pass
    
    def apply_sisr(self, degraded_image: np.ndarray):
        pass