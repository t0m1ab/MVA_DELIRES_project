import os
from pathlib import Path
from logging import Logger, getLogger
import numpy as np
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from delires.methods.diffuser import Diffuser
from delires.methods.dps.dps_configs import DPSConfig, DPSDeblurConfig, DPSSchedulerConfig
from delires.methods.dps.dps_deblur import apply_DPS_for_deblurring

from delires.params import (
    MODELS_PATH,
    OPERATORS_PATH,
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
        self.load_model(config, scheduler_config=DPSSchedulerConfig) # load and store self.model and self.scheduler

        # SISR
        self.classical_degradation = getattr(config, "sisr_classical_degradation", False)
        self.sf: int = getattr(config, "sf", 4)

        # Deblurring
        self.kernel_filename: str = None
        self.kernel: torch.Tensor = None
     
    def apply_debluring(
            self,
            config: DPSDeblurConfig,
            clean_image_filename: str,
            degraded_image_filename: str,
            degraded_dataset_name: str = None,
            experiment_name: str = None,
            kernel_filename: str = None,
            use_png_data: bool = True,
            img_ext: str = "png",
            save: bool = False,
        ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Apply DPS deblurring to a given degraded image.

        ARGUMENTS:
            - config: DPSDeblurConfig used for the deblurring.
            - clean_image_filename: name of the clean image (without extension).
            - degraded_image_filename: name of the degraded image (without extension).
            - degraded_dataset_name: name of the degraded dataset (potential subfolder in DEGRADED_DATA_PATH).
            - experiment_name: name of the experiment (potential subfolder in RESTORED_DATA_PATH). If None, then save directly in RESTORED_DATA_PATH.
            - kernel_filename: name of the kernel (without extension). If None, then try to use self.kernel and self.kernel_filename.
            - use_png_data: if True, the degraded image will be loaded from PNG file (=> uint values => [0,1] clipping) otherwise from npy file (=> float values can be unclipped).
            - img_ext: extension of the images (default: "png").
            - save: if True, the restored image will be saved in the RESTORED_DATA_PATH/<experiment_name> folder.
        
        RETURNS:
            - restored_image: np.ndarray containing the restored image.
            - metrics: dict {metric_name: metric_value} containing the metrics of the deblurring.
        """

        if self.model is None or self.scheduler is None:
            raise ValueError("The model and scheduler objects must be loaded before applying deblurring.")

        # load images (and kernel if specified)        
        clean_image, degraded_image = self.load_data(
            degraded_dataset_name=degraded_dataset_name if degraded_dataset_name is not None else "",
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            kernel_filename=kernel_filename,
            use_png_data=use_png_data,
            img_ext=img_ext,
        )

        # apply DPS deblurring
        self.log_banner("DPS Deblurring")
        self.logger.info(f"- model_name: {self.config.model_name}")
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

        self.log_banner("--------------") # separate logs between different images

        return restored_image, metrics
    
    def apply_sisr(self, degraded_image: np.ndarray):
        raise NotImplementedError


def main():

    ### DEMO DPS deblurring
    
    # setup device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()

    dps_config = DPSConfig()
    dps_diffuser = DPSDiffuser(dps_config, autolog="dps_debluring_test", device=device)

    # dps_diffuser.load_blur_kernel("gaussian_kernel_05")
    dps_diffuser.load_blur_kernel("motion_kernel_example")

    dps_deblur_config = DPSDeblurConfig()
    img_name = "69037"
    _ = dps_diffuser.apply_debluring(
        config=dps_deblur_config,
        clean_image_filename=img_name,
        degraded_image_filename=img_name,
        degraded_dataset_name="blurred_dataset",
        # kernel_filename="gaussian_kernel_05",
        save=True,
    )


if __name__ == "__main__":
    main()