import os
import numpy as np
from pathlib import Path
from logging import Logger, getLogger
from typing import Union
from abc import abstractmethod
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from delires.data import load_operator
from delires.utils import utils_image
from delires.utils.utils_logger import logger_info
from delires.methods.dps.dps_configs import DPSConfig, DPSSchedulerConfig
from delires.methods.pigdm.pigdm_configs import PiGDMConfig, PiGDMSchedulerConfig
from delires.methods.diffpir.guided_diffusion.unet import UNetModel
from delires.methods.diffpir.guided_diffusion.respace import SpacedDiffusion
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
    OPERATORS_PATH,
)


class Diffuser():

    def __init__(self, logger: Logger = None, autolog: str = None, device = "cpu"):
        
        # main attributes
        self.device = device
        self.logger = logger
        if autolog is not None and self.logger is None: # create a logger if not provided but if autolog is specified
            Path(RESTORED_DATA_PATH).mkdir(parents=True, exist_ok=True)
            logger_info(autolog, log_path=os.path.join(RESTORED_DATA_PATH, f"{autolog}.log"))
            self.logger = getLogger(autolog)

        # diffusion model attributes
        self.config: DPSConfig | PiGDMConfig | DiffPIRConfig = None
        self.model: UNet2DModel | UNetModel = None # UNet2DModel (huggingface) | UNetModel (diffpir) object
        self.scheduler: DDPMScheduler = None # DDPMScheduler (huggingface) object
        self.diffusion: SpacedDiffusion = None # SpacedDiffusion (diffpir) object

        # deblurring attributes
        self.kernel_filename: str = None
        self.kernel: np.ndarray = None

        # inpainting attributes
        self.masks_filename: str = None
        self.mask_index: int = None
        self.mask: np.ndarray = None
        
    def load_blur_kernel(self, kernel_filename: str = None, kernel_family: str = None, kernel_idx: str | int = None):
        """ Load a blur kernel from a file using the given information. """
        k = load_operator(filename=kernel_filename, operator_family=kernel_family, operator_idx=kernel_idx) # load kernel
        self.kernel = np.expand_dims(k, axis=(0,1)) # add batch dim and channel dim for compatibility
        self.kernel_filename = kernel_filename if kernel_filename is not None else f"{kernel_family}_{kernel_idx}"

    def load_inpainting_mask(self, mask_filename: str = None, mask_family: str = None, mask_idx: str | int = None):
        """ Load a mask from a file using the given information. """
        mask = load_operator(filename=mask_filename, operator_family=mask_family, operator_idx=mask_idx) # load masks
        self.mask = np.expand_dims(mask, axis=-1) # add channel dim for compatibility
        self.mask_filename = mask_filename
    
    def load_downsample_kernel(self, k_index: int = 0, cwd: str = ""):
        """ Load a downsampling kernel from a file or from a given kernel filename (name without extension). """
        raise NotImplementedError("The degradation method is not implemented yet in children classes.")
    
    def load_model(
            self, 
            config: DPSConfig | PiGDMConfig | DiffPIRConfig, 
            scheduler_config: DPSSchedulerConfig | PiGDMSchedulerConfig | None = None,
        ) -> None:
        """ Load the model and diffusion objects from the given config. """

        if config.model_name in DIFFPIR_NETWOKRS: # load UNetModel nn from diffpir code
            if self.logger is not None:
                self.logger.info(f"Loading DiffPIR network '{config.model_name}' from: {MODELS_PATH}")
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
            if self.logger is not None:
                self.logger.info(f"Loading HuggingFace network: {config.model_name}")
            ddpm = DDPMPipeline.from_pretrained(config.model_name)
            self.model = ddpm.unet
            self.scheduler = ddpm.scheduler

    def load_image_data(
            self,
            degraded_dataset_name: str,
            clean_image_filename: str,
            degraded_image_filename: str,
            use_png_data: bool,
            img_ext: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load clean and degraded images from files and return them as torch tensors.
        If use_png_data is True, the degraded image will be loaded from PNG file (=> uint values => [0,1] clipping)
        otherwise from npy file (=> float values can be unclipped).
        """

        # load clean image as float32 tensor of shape (1,C,W,H)
        clean_image_path = os.path.join(CLEAN_DATA_PATH, f"{clean_image_filename}.{img_ext}")
        clean_image = utils_image.uint2float32(utils_image.imread_uint(clean_image_path))

        # load degraded image as float32 tensor of shape (1,C,W,H)
        if use_png_data: # load from PNG file (=> uint values => [0,1] clipping)
            degraded_image_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, f"{degraded_image_filename}.png")
            degraded_image = utils_image.uint2float32(utils_image.imread_uint(degraded_image_path))
        else: # load from npy file (=> float values can be unclipped)
            degraded_image_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, f"{degraded_image_filename}.npy")
            degraded_image = torch.tensor(np.load(degraded_image_path), dtype=torch.float32)
        
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