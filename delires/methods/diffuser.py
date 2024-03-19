import os
import numpy as np
from pathlib import Path
from logging import Logger, getLogger
from typing import Union
from abc import abstractmethod
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from delires.data import load_downsample_kernel, load_blur_kernel
from delires.utils import utils_image
from delires.utils.utils_logger import logger_info
from delires.methods.dps.dps_configs import DPSConfig, DPSSchedulerConfig
from delires.methods.pigdm.pigdm_configs import PiGDMConfig, PiGDMSchedulerConfig
from delires.methods.diffpir.diffpir_configs import DiffPIRConfig
from delires.methods.diffpir.utils import utils_model
from delires.methods.diffpir.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from delires.params import (
    CLEAN_DATA_PATH,
    DEGRADED_DATA_PATH,
    RESTORED_DATA_PATH,
    DIFFPIR_NETWOKRS,
    MODELS_PATH,
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
        
    def load_blur_kernel(self, kernel_filename: str|None = None):
        """ Load a blur kernel from a file or from a given kernel filename (name without extension). """
        np_kernel = load_blur_kernel(kernel_filename)
        # add batch dim and channel dim for compatibility and set as tensor
        self.kernel = torch.tensor(np.expand_dims(np_kernel, axis=(0,1)), dtype=torch.float32)
        self.kernel_filename = kernel_filename
    
    def load_model(
            self, 
            config: DPSConfig | PiGDMConfig | DiffPIRConfig, 
            scheduler_config: DPSSchedulerConfig | PiGDMSchedulerConfig | None = None,
        ) -> None:
        """ Load the model and diffusion objects from the given config. """

        if config.model_name in DIFFPIR_NETWOKRS: # load UNetModel nn from diffpir code
            print(f"Loading DiffPIR network: {config.model_name}")
            model_path = os.path.join(MODELS_PATH, f"{config.model_name}.pt")
            if config.model_name == DIFFPIR_NETWOKRS[0]: # diffusion_ffhq_10m
                model_config = dict(
                    model_path=model_path,
                    num_channels=128,
                    num_res_blocks=1,
                    attention_resolutions="16",
                )
            elif config.model_name == DIFFPIR_NETWOKRS[1]: # 256x256_diffusion_uncond
                model_config = dict(
                    model_path=model_path,
                    num_channels=256,
                    num_res_blocks=2,
                    attention_resolutions="8,16,32",
                )
            else:
                raise KeyError(f"A new diffpir network was added to DIFFPIR_NETWOKRS but is not handled in the {self}.load_model method: {config.model_name}")
            args = utils_model.create_argparser(model_config).parse_args([])
            # load model and diffusion objects but don't need diffusion so it is discarded
            model, _ = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
            model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
            self.model = model
            self.scheduler = DDPMScheduler.from_config(config=scheduler_config().__dict__)

        else: # load DDPMPipeline model from HuggingFace
            print(f"Loading HuggingFace network: {config.model_name}")
            ddpm = DDPMPipeline.from_pretrained(config.model_name)
            self.model = ddpm.unet
            self.scheduler = ddpm.scheduler

    def load_data(
            self,
            degraded_dataset_name: str,
            clean_image_filename: str,
            degraded_image_filename: str,
            kernel_filename: str,
            use_png_data: bool,
            img_ext: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load clean and degraded images (and blur kernel if specified) from files and return them as torch tensors.
        If use_png_data is True, the degraded image will be loaded from PNG file (=> uint values => [0,1] clipping)
        otherwise from npy file (=> float values can be unclipped).
        """

        # load CLEAN image: float32 tensor of shape (1,C,W,H)
        clean_image_path = os.path.join(CLEAN_DATA_PATH, f"{clean_image_filename}.{img_ext}")
        clean_image = utils_image.uint2float32(utils_image.imread_uint(clean_image_path))

        # load DEGRADED image: float32 tensor of shape (1,C,W,H)
        if use_png_data: # load from PNG file (=> uint values => [0,1] clipping)
            degraded_image_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, f"{degraded_image_filename}.png")
            degraded_image = utils_image.uint2float32(utils_image.imread_uint(degraded_image_path))
        else: # load from npy file (=> float values can be unclipped)
            degraded_image_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, f"{degraded_image_filename}.npy")
            degraded_image = torch.tensor(np.load(degraded_image_path), dtype=torch.float32)

        # load KERNEL if necessary (otherwise use self.kernel and self.kernel_filename): float32 tensor of shape (1,1,K,K)
        if kernel_filename is not None:
            self.load_blur_kernel(kernel_filename)

        if self.kernel is None or self.kernel_filename is None:
            raise ValueError("The blur kernel must be loaded before applying deblurring.")
        
        return clean_image, degraded_image

    def save_restored_image(
            self, 
            restored_image: np.ndarray, 
            restored_image_filename: str,
            path: str = None,
            img_ext: str = "png",
        ):
        path = path if path is not None else RESTORED_DATA_PATH
        Path(path).mkdir(parents=True, exist_ok=True)
        restored_image_path = os.path.join(path, f"{restored_image_filename}.{img_ext}")
        utils_image.imsave(restored_image, restored_image_path)
        if self.logger is not None:
            self.logger.info(f"Restoration saved as: {restored_image_path}")
        
    def log_banner(self, msg: str, width: int = 20):
        if self.logger is not None:
            self.logger.info(width*"-" + f" {msg} " + width*"-")
        else:
            print(width*"-" + f" {msg} " + width*"-")

    @abstractmethod
    def apply_debluring(self, degraded_image: np.ndarray):
        raise NotImplementedError
    
    @abstractmethod
    def apply_sisr(self, degraded_image: np.ndarray):
        raise NotImplementedError