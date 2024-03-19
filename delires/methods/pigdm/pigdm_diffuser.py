import os
from pathlib import Path
from logging import Logger, getLogger
import numpy as np
import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from delires.data import load_downsample_kernel
from delires.methods.diffuser import Diffuser
from delires.methods.pigdm.pigdm_configs import PiGDMConfig, PiGDMDeblurConfig, PiGDMInpaintingConfig, PiGDMSchedulerConfig
from delires.methods.diffpir.utils import utils_model, utils_image
from delires.methods.pigdm.pigdm_deblur import apply_PiGDM_for_deblurring
from delires.methods.pigdm.pigdm_inpainting import apply_PiGDM_for_inpainting

from delires.params import (
    MODELS_PATH,
    OPERATORS_PATH,
    CLEAN_DATA_PATH,
    DEGRADED_DATA_PATH,    
    RESTORED_DATA_PATH,
    DIFFPIR_NETWOKRS
)


class PiGDMDiffuser(Diffuser):

    def __init__(self, config: PiGDMConfig, logger: Logger = None, autolog: str = None, device = "cpu"):
        super().__init__(device=device, logger=logger, autolog=autolog)

        self.config = config
    
        self.model: UNet2DModel = None # torch.nn.Module object
        self.scheduler: DDPMScheduler = None # ddpmscheduler object
        self.load_model(config, scheduler_config=PiGDMSchedulerConfig) # store in self.model and self.diffusion

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
        
    def load_mask(self, masks_filename: str, mask_index: int = 0):
        """ Load a mask from a file with a given mask set filename and given index within the selected masks set (name without extension). """
        masks = np.load(os.path.join(OPERATORS_PATH, f"{masks_filename}.npy"))
        self.mask = masks[mask_index]
        self.masks_filename = masks_filename
        self.mask_index = mask_index
        
    def apply_debluring(
            self,
            config: PiGDMDeblurConfig,
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
        Apply PiGDM deblurring to a given degraded image.
        Apply PiGDM deblurring to a given degraded image.

        ARGUMENTS:
            - config: PiGDMDeblurConfig used for the deblurring.
            - config: PiGDMDeblurConfig used for the deblurring.
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

        # check if model and scheduler are loaded
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

        # apply PiGDM deblurring
        self.log_banner("PiGDM Deblurring")
        self.logger.info(f"- model_name: {self.config.model_name}")
        restored_image, metrics = apply_PiGDM_for_deblurring(
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
            device=self.device,
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
     
    def apply_inpainting(
            self,
            config: PiGDMInpaintingConfig,
            clean_image_filename: str,
            degraded_image_filename: str,
            degraded_dataset_name: str = None,
            experiment_name: str = None,
            masks_filename: str = None,
            mask_index: int = 0,
            img_ext: str = "png",
            save: bool = False,
        ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Apply PiGDM inpainting to a given degraded image.

        ARGUMENTS:
            - config: PiGDMInpaintingConfig used for the inpainting.
            - clean_image_filename: name of the clean image (without extension).
            - degraded_image_filename: name of the degraded image (without extension).
            - degraded_dataset_name: name of the degraded dataset (potential subfolder in DEGRADED_DATA_PATH).
            - experiment_name: name of the experiment (potential subfolder in RESTORED_DATA_PATH). If None, then save directly in RESTORED_DATA_PATH.
            - masks_filename: the name of the set of masks (without extension, ex: "my_masks"). If None, then try to use self.mask and self.masks filename.
            - mask_index: the index of the mask in the set of masks. If None, then try to use self.mask, self.masks filename and self.mask_idx.
            - img_ext: extension of the images (default: "png").
            - save: if True, the restored image will be saved in the RESTORED_DATA_PATH/<experiment_name> folder.
        
        RETURNS:
            - restored_image: np.ndarray containing the restored image.
            - metrics: dict {metric_name: metric_value} containing the metrics of the inpainting.
        """

        if self.model is None or self.scheduler is None:
            raise ValueError("The model and scheduler objects must be loaded before applying inpainting.")

        # load images
        degraded_dataset_name = degraded_dataset_name if degraded_dataset_name is not None else ""

        clean_image_png_path = os.path.join(CLEAN_DATA_PATH, f"{clean_image_filename}.{img_ext}")
        degraded_image_png_path = os.path.join(DEGRADED_DATA_PATH, degraded_dataset_name, f"{degraded_image_filename}.png")

        clean_image = utils_image.imread_uint(clean_image_png_path)
        degraded_image = utils_image.imread_uint(degraded_image_png_path)

        # load kernel if necessary (otherwise use self.mask and self.masks_filename)
        if masks_filename is not None:
            self.load_mask(masks_filename, mask_index)

        if (self.masks_filename is None and self.mask is not None) or (masks_filename is not None and self.mask is None):
            raise ValueError("To load a mask, please indicate both the masks set and the mask index.")
        if self.mask is None or self.masks_filename is None:
            raise ValueError("The mask must be loaded before applying inpainting.")

        # apply PiGDM inpainting
        restored_image, metrics = apply_PiGDM_for_inpainting(
            config=config,
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            masks_filename=self.masks_filename,
            mask_index=self.mask_index,
            clean_image=clean_image,
            degraded_image=degraded_image,
            mask=self.mask,
            model=self.model,
            scheduler=self.scheduler,
            logger=self.logger,
            device=self.device
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

        self.log_banner("----------------") # separate logs between different images

        return restored_image, metrics
    
    def apply_sisr(self, degraded_image: np.ndarray):
        raise NotImplementedError("The apply_sisr method is not implemented yet for the PiGDMDiffuser class.")


def main():

    # setup device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()

    ### DEMO PiGDM deblurring

    pigdm_config = PiGDMConfig()
    pigdm_diffuser = PiGDMDiffuser(pigdm_config, autolog="pigdm_debluring_test", device=device)

    # pigdm_diffuser.load_blur_kernel("gaussian_kernel_05")
    pigdm_diffuser.load_blur_kernel("motion_kernel_example")

    pigdm_deblur_config = PiGDMDeblurConfig()
    img_name = "1"
    _ = pigdm_diffuser.apply_debluring(
        config=pigdm_deblur_config,
        clean_image_filename=img_name,
        degraded_image_filename=img_name,
        degraded_dataset_name="blurred_ffhq_test20",
        # kernel_filename="gaussian_kernel_05",
        save=True, # we save the image on the fly for the demo
    )
    
    ### DEMO PiGDM inpainting

    # pigdm_config = PiGDMConfig()
    # pigdm_diffuser = PiGDMDiffuser(pigdm_config, autolog="pigdm_inpainting_test", device=device)

    # # pigdm_diffuser.load_blur_kernel("gaussian_kernel_05")
    # pigdm_diffuser.load_blur_kernel("motion_kernel_example")
    # masks_name = "box_masks"
    # mask_index = 0
    # pigdm_diffuser.load_mask(masks_name, mask_index)  

    # pigdm_inpaint_config = PiGDMInpaintingConfig()
    # img_name = "0"
    # _ = pigdm_diffuser.apply_inpainting(
    #     config=pigdm_inpaint_config,
    #     clean_image_filename=img_name,
    #     degraded_image_filename=img_name,
    #     degraded_dataset_name="masked_dataset",
    #     masks_filename=masks_name,
    #     mask_index=mask_index,
    #     save=True, # we save the image on the fly for the demo
    #     degraded_dataset_name="blurred_dataset",
    #     # kernel_filename="gaussian_kernel_05",
    #     save=True,
    # )


if __name__ == "__main__":
    main()