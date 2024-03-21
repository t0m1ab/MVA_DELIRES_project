import os
from pathlib import Path
from logging import Logger
import argparse
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning) 
import torch

from delires.methods.diffuser import Diffuser
from delires.methods.pigdm.pigdm_configs import PiGDMConfig, PiGDMDeblurConfig, PiGDMInpaintingConfig, PiGDMSchedulerConfig
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
        self.load_model(config, scheduler_config=PiGDMSchedulerConfig)
        
    def apply_debluring(
            self,
            config: PiGDMDeblurConfig,
            clean_image_filename: str,
            degraded_image_filename: str,
            degraded_dataset_name: str = None,
            experiment_name: str = None,
            kernel_filename: str = None,
            kernel_family: str = None,
            kernel_idx: str | int = None,
            use_png_data: bool = True,
            img_ext: str = "png",
            save: bool = False,
        ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Apply PiGDM deblurring to a given degraded image.

        ARGUMENTS:
            - config: PiGDMDeblurConfig used for the deblurring.
            - config: PiGDMDeblurConfig used for the deblurring.
            - clean_image_filename: name of the clean image (without extension).
            - degraded_image_filename: name of the degraded image (without extension).
            - degraded_dataset_name: name of the degraded dataset (potential subfolder in DEGRADED_DATA_PATH).
            - experiment_name: name of the experiment (potential subfolder in RESTORED_DATA_PATH). If None, then save directly in RESTORED_DATA_PATH.
            - kernel_filename: name of the kernel (without extension). If None, then try to use self.kernel and self.kernel_filename.
            - kernel_family: name of the kernel family which is a potential subfolder in OPERATORS_PATH (ex: "levin09").
            - kernel_idx: index of the kernel in the family (ex: 0).
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
        clean_image, degraded_image = self.load_image_data(
            degraded_dataset_name=degraded_dataset_name if degraded_dataset_name is not None else "",
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            use_png_data=use_png_data,
            img_ext=img_ext,
        )

        # load kernel if necessary (otherwise use self.kernel and self.kernel_filename) as float32 tensor of shape (1,1,K,K)
        if kernel_filename is not None:
            self.load_blur_kernel(kernel_filename=kernel_filename, kernel_family=kernel_family, kernel_idx=kernel_idx)

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

        self.log_banner("----------------")

        return restored_image, metrics
     
    def apply_inpainting(
            self,
            config: PiGDMInpaintingConfig,
            clean_image_filename: str,
            degraded_image_filename: str,
            degraded_dataset_name: str = None,
            experiment_name: str = None,
            mask_filename: str = None,
            mask_family: str = None,
            mask_idx: str | int = None,
            use_png_data: bool = True,
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
            - mask_filename: name of the mask (without extension). If None, then try to use self.kernel and self.kernel_filename.
            - mask_family: name of the mask family which is a potential subfolder in OPERATORS_PATH (ex: "box_masks").
            - mask_idx: index of the mask in the family (ex: 0).
            - use_png_data: if True, the degraded image will be loaded from PNG file (=> uint values => [0,1] clipping) otherwise from npy file (=> float values can be unclipped).
            - img_ext: extension of the images (default: "png").
            - save: if True, the restored image will be saved in the RESTORED_DATA_PATH/<experiment_name> folder.
        
        RETURNS:
            - restored_image: np.ndarray containing the restored image.
            - metrics: dict {metric_name: metric_value} containing the metrics of the inpainting.
        """

        if self.model is None or self.scheduler is None:
            raise ValueError("The model and scheduler objects must be loaded before applying inpainting.")

        # load images (and kernel if specified)        
        clean_image, degraded_image = self.load_image_data(
            degraded_dataset_name=degraded_dataset_name if degraded_dataset_name is not None else "",
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            use_png_data=use_png_data,
            img_ext=img_ext,
        )

        # load kernel if necessary (otherwise use self.kernel and self.kernel_filename)
        if mask_filename is not None:
            self.load_inpainting_mask(mask_filename=mask_filename, mask_family=mask_family, mask_idx=mask_idx)

        if self.mask is None:
            raise ValueError("The mask must be loaded before applying inpainting.")

        # apply PiGDM inpainting
        self.log_banner("PiGDM Inpainting")
        self.logger.info(f"- model_name: {self.config.model_name}")
        restored_image, metrics = apply_PiGDM_for_inpainting(
            config=config,
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            masks_filename=self.masks_filename,
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
        
    def demo_pigdm_deblur(device: torch.device, filename: str = None):
        """ Demonstration of PiGDM deblurring. """

        pigdm_config = PiGDMConfig()
        pigdm_diffuser = PiGDMDiffuser(pigdm_config, autolog="pigdm_debluring_test", device=device)

        pigdm_diffuser.load_blur_kernel(
            kernel_filename=None,
            kernel_family="levin09",
            kernel_idx=1,
        )

        img_name = "1"

        _ = pigdm_diffuser.apply_debluring(
            config=PiGDMDeblurConfig(),
            clean_image_filename=img_name,
            degraded_image_filename=img_name if filename is None else filename,
            degraded_dataset_name="blurred_ffhq_test20",
            # kernel_filename="gaussian_kernel_05",
            save=True, # we save the image on the fly for the demo
        )
            
    def demo_pigdm_inpainting(device: torch.device, filename: str = None):
        """ Demonstration of PiGDM inpainting. """

        pigdm_config = PiGDMConfig()
        pigdm_diffuser = PiGDMDiffuser(pigdm_config, autolog="pigdm_inpainting_test", device=device)

        pigdm_diffuser.load_inpainting_mask(
            mask_family="box_masks", 
            mask_idx=1,
        )  

        img_name = "1"

        _ = pigdm_diffuser.apply_inpainting(
            config=PiGDMInpaintingConfig(),
            clean_image_filename=img_name,
            degraded_image_filename=img_name if filename is None else filename,
            degraded_dataset_name="masked_ffhq_test20",
            # mask_family=mask_family,
            # mask_idx=mask_idx,
            save=True,
        )
    
    parser = argparse.ArgumentParser(description="Demonstration of PiGDM methods.")
    parser.add_argument(
        "--debluring",
        "-d",
        action="store_true", 
        help="demonstration of PiGDM debluring."
    )
    parser.add_argument(
        "--inpainting",
        "-i",
        action="store_true",
        help="demonstration of PiGDM inpainting."
    )
    parser.add_argument(
        "--image",
        "-x",
        dest="filename",
        type=str,
        default=None,
        help="name of the image to process (without extension)."
    )

    args = parser.parse_args()

    if args.debluring and args.inpainting:
        raise ValueError("Please choose only one demonstration at a time.")
    elif not args.debluring and not args.inpainting:
        _ = PiGDMDiffuser(PiGDMConfig(), device=device)
    elif args.debluring:
        demo_pigdm_deblur(device, filename=args.filename)
    elif args.inpainting:
        demo_pigdm_inpainting(device, filename=args.filename)


if __name__ == "__main__":
    main()
    # example command 1: python pigdm_diffuser.py -d -x 4
    # example command 2: python pigdm_diffuser.py -i -x 8