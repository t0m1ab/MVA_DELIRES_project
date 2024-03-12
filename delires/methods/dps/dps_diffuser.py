import os
from pathlib import Path
from logging import Logger, getLogger
import numpy as np
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from delires.data import load_downsample_kernel, load_blur_kernel
from delires.utils.utils_logger import logger_info
from delires.utils import utils_image
from delires.methods.diffuser import Diffuser
from delires.methods.dps.dps_configs import DPSConfig, DPSDeblurConfig, DPSSchedulerConfig
from delires.methods.diffpir.utils import utils_model
from delires.methods.dps.dps_deblur import apply_DPS_for_deblurring

from delires.methods.diffpir.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from delires.params import (
    MODELS_PATH,
    KERNELS_PATH,
    CLEAN_DATA_PATH,
    DEGRADED_DATA_PATH,    
    RESTORED_DATA_PATH,
    DIFFPIR_NETWOKRS
)
# DDPMPipeline.from_pretrained(model_name)

class DPSDiffuser(Diffuser):

    def __init__(self, config: DPSConfig, logger: Logger = None, autolog: str = None, device = "cpu"):
        super().__init__(device=device, logger=logger, autolog=autolog)

        self.config = config
    
        self.model: UNet2DModel = None # torch.nn.Module object
        self.scheduler: DDPMScheduler = None # ddpmscheduler object
        self.load_model(config) # load and store self.model and self.scheduler

        # SISR
        self.classical_degradation = getattr(config, "sisr_classical_degradation", False)
        self.sf: int = getattr(config, "sf", 4)

        # Deblurring
        self.kernel_filename: str = None
        self.kernel: np.ndarray = None

    def load_downsample_kernel(
        self,
        k_index: int = 0,
        cwd: str = "",
        ):
        self.kernel = load_downsample_kernel(self.classical_degradation, self.sf, k_index, cwd)
        
    def load_blur_kernel(self, kernel_filename: str|None = None):
        """ Load a blur kernel from a file or from a given kernel filename (name without extension). """
        self.kernel = load_blur_kernel(kernel_filename)
        self.kernel_filename = kernel_filename
    
    def load_model(self, config: DPSConfig) -> None:
        """ Load the model from the given config. """
        
        if config.model_name in DIFFPIR_NETWOKRS: # load UNetModel nn from diffpir code
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

        else: # load DDPMPipeline model from HuggingFace
            self.model = DDPMPipeline.from_pretrained(config.model_name).unet
        
        self.scheduler = DDPMScheduler.from_config(config=DPSSchedulerConfig().__dict__)

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
            self.logger.info(f"Restored image saved in: {restored_image_path}")
        
    def apply_debluring(
            self,
            config: DPSDeblurConfig,
            clean_image_filename: str,
            degraded_image_filename: str,
            degraded_dataset_name: str = None,
            experiment_name: str = None,
            kernel_filename: str = None,
            img_ext: str = "png",
            save: bool = False,
        ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Apply DiffPIR deblurring to a given degraded image.

        ARGUMENTS:
            - config: DiffPIRDeblurConfig used for the deblurring.
            - clean_image_filename: name of the clean image (without extension).
            - degraded_image_filename: name of the degraded image (without extension).
            - degraded_dataset_name: name of the degraded dataset (potential subfolder in DEGRADED_DATA_PATH).
            - experiment_name: name of the experiment (potential subfolder in RESTORED_DATA_PATH). If None, then save directly in RESTORED_DATA_PATH.
            - kernel_filename: name of the kernel (without extension). If None, then try to use self.kernel and self.kernel_filename.
            - img_ext: extension of the images (default: "png").
            - save: if True, the restored image will be saved in the RESTORED_DATA_PATH/<experiment_name> folder.
        
        RETURNS:
            - restored_image: np.ndarray containing the restored image.
            - metrics: dict {metric_name: metric_value} containing the metrics of the deblurring.
        """

        if self.model is None or self.scheduler is None:
            raise ValueError("The model and scheduler objects must be loaded before applying deblurring.")

        # load images
        degraded_dataset_name = degraded_dataset_name if degraded_dataset_name is not None else ""
        clean_image_png_path = os.path.join(CLEAN_DATA_PATH, f"{clean_image_filename}.{img_ext}")
        degraded_image_png_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, f"{degraded_image_filename}.png")

        clean_image = utils_image.imread_uint(clean_image_png_path)
        degraded_image = utils_image.imread_uint(degraded_image_png_path)

        # load kernel if necessary (otherwise use self.kernel and self.kernel_filename)
        if kernel_filename is not None:
            self.load_blur_kernel(kernel_filename)

        if self.kernel is None or self.kernel_filename is None:
            raise ValueError("The blur kernel must be loaded before applying deblurring.")

        # apply DiffPIR deblurring
        restored_image, metrics = apply_DPS_for_deblurring(
            config=config,
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            kernel_filename=self.kernel_filename,
            clean_image=clean_image,
            degraded_image=degraded_image,
            kernel=self.kernel,
            model=self.model,
            scheduler=self.scheduler,
            logger=self.logger,
            device = self.device
        )

        # save restored image
        if save:
            experiment_name = experiment_name if experiment_name is not None else ""
            self.save_restored_image(
                restored_image=restored_image,
                restored_image_filename=degraded_image_filename,
                path=os.path.join(RESTORED_DATA_PATH, experiment_name),
                img_ext=img_ext,
            )

        self.logger.info(50*"-") # separate logs between different images

        return restored_image, metrics
    
    def apply_sisr(self, degraded_image: np.ndarray):
        pass


def main():

    # quick demo of the DPS deblurring
    
    # setup device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()

    dps_config = DPSConfig()
    dps_diffuser = DPSDiffuser(dps_config, autolog="dps_debluring_test", device=device)

    dps_diffuser.load_blur_kernel("gaussian_kernel_05")
    # dps_diffuser.load_blur_kernel("motion_kernel_1")

    dps_deblur_config = DPSDeblurConfig()
    img_name = "theilo"
    _ = dps_diffuser.apply_debluring(
        config=dps_deblur_config,
        clean_image_filename=img_name,
        degraded_image_filename=img_name,
        degraded_dataset_name="blurred_dataset",
        kernel_filename="gaussian_kernel_05",
        save=True,
    )


if __name__ == "__main__":
    main()