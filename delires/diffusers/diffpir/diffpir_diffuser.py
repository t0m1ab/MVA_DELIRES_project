import os
from pathlib import Path
import numpy as np

from delires.data import load_downsample_kernel, load_blur_kernel
from delires.diffusers.diffuser import Diffuser
from delires.diffusers.diffpir.my_ddpir_deblur import DiffPIRDeblurConfig, method_apply_DiffPIR_for_deblurring


def load_image(filename: str) -> np.ndarray:
    raise NotImplementedError


class DiffPIRDiffuser(Diffuser):

    def __init__(self, device = "cpu", **kwargs):
        self.classical_degradation = kwargs.get("sisr_classical_degradation", False)
        self.sf = kwargs.get("sf", 4)
        self.device = device

        self.model, self.diffusion = self.load_model(kwargs.get("model_path"))

        self.kernel_filename = None
        self.kernel = None
        self.kernel_4d = None

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
        self.kernel_filename = None
        raise NotImplementedError, "store the kernel filename ? (with or without extension ?)"
        self.kernel, self.kernel_4d = load_blur_kernel(diy_kernel_path, self.device, cwd)
    
    def load_model(self, model_path: str):
         raise NotImplementedError, "reproduce lines 180-202 in my_ddpir_deblur.py (should be the same for inpainting, SISR ?)"

    def save_restored(self, path: str):
        pass
        
    def apply_debluring(
            self,
            config: DiffPIRDeblurConfig,
            clean_image_filename: str,
            degraded_image_filename: str,
        ):

        clean_image = load_image(clean_image_filename)
        degraded_image = load_image(degraded_image_filename)

        # TODO: return the restored image and use method DiffPIRDiffuser.save_restored
        method_apply_DiffPIR_for_deblurring(
            config=config,
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            kernel_filename=self.kernel_filename,
            ext=".png",
            clean_image=clean_image,
            degraded_image=degraded_image,
            kernel=self.kernel,
            k_4d=self.kernel_4d,
            model=self.model,
            diffusion=self.diffusion,
        )
    
    def apply_sisr(self, degraded_image: np.ndarray):
        pass