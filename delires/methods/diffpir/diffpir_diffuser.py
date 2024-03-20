import os
from pathlib import Path
from logging import Logger
import argparse
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning) 
import torch

from delires.methods.diffuser import Diffuser
from delires.methods.diffpir.diffpir_configs import DiffPIRConfig, DiffPIRDeblurConfig, DiffPIRInpaintingConfig
from delires.methods.diffpir.utils import utils_model
from delires.methods.diffpir.diffpir_deblur import apply_DiffPIR_for_deblurring
from delires.methods.diffpir.diffpir_inpainting import apply_DiffPIR_for_inpainting

from delires.methods.diffpir.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from delires.params import (
    MODELS_PATH,
    OPERATORS_PATH,
    CLEAN_DATA_PATH,
    DEGRADED_DATA_PATH,    
    RESTORED_DATA_PATH,
    DIFFPIR_NETWOKRS,
)


class DiffPIRDiffuser(Diffuser):

    def __init__(self, config: DiffPIRConfig, logger: Logger = None, autolog: str = None, device = "cpu"):
        super().__init__(device=device, logger=logger, autolog=autolog)
        self.config = config
        self.load_model(config, scheduler_config=None)
    
    def load_model(self, config: DiffPIRConfig, scheduler_config: None = None) -> None:
        """ Load the model and diffusion objects from the given config. """

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
        model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

        self.model = model
        self.diffusion = diffusion

    def apply_debluring(
            self,
            config: DiffPIRDeblurConfig,
            clean_image_filename: str,
            degraded_image_filename: str,
            degraded_dataset_name: str = None,
            experiment_name: str = None,
            kernel_filename: str = None,
            kernel_family: str = None,
            kernel_idx: str | int = None,
            use_png_data: bool = False,
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
            - kernel_family: name of the kernel family which is a potential subfolder in OPERATORS_PATH (ex: "levin09").
            - kernel_idx: index of the kernel in the family (ex: 0).
            - use_png_data: if True, the degraded image will be loaded from PNG file (=> uint values => [0,1] clipping) otherwise from npy file (=> float values can be unclipped).
            - img_ext: extension of the images (default: "png").
            - save: if True, the restored image will be saved in the RESTORED_DATA_PATH/<experiment_name> folder.
        
        RETURNS:
            - restored_image: np.ndarray containing the restored image.
            - metrics: dict {metric_name: metric_value} containing the metrics of the deblurring.
        """

        if self.model is None or self.diffusion is None:
            raise ValueError("The model and diffusion objects must be loaded before applying deblurring.")

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

        # apply DiffPIR deblurring
        self.log_banner("DiffPIR Deblurring")
        self.logger.info(f"- model_name: {self.config.model_name}")
        restored_image, metrics = apply_DiffPIR_for_deblurring(
            config=config,
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            kernel_filename=self.kernel_filename,
            clean_image=clean_image,
            degraded_image=degraded_image,
            kernel=self.kernel,
            model=self.model,
            diffusion=self.diffusion,
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

        self.log_banner("------------------")

        return restored_image, metrics
        
    def apply_inpainting(
            self,
            config: DiffPIRInpaintingConfig,
            clean_image_filename: str,
            degraded_image_filename: str,
            degraded_dataset_name: str = None,
            experiment_name: str = None,
            masks_filename: str = None,
            mask_index: int = 0,
            use_png_data: bool = False,
            img_ext: str = "png",
            save: bool = False,
        ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Apply DiffPIR inpainting to a given degraded image.

        ARGUMENTS:
            - config: DiffPIRInpaintingConfig used for the inpainting.
            - clean_image_filename: name of the clean image (without extension).
            - degraded_image_filename: name of the degraded image (without extension).
            - degraded_dataset_name: name of the degraded dataset (potential subfolder in DEGRADED_DATA_PATH).
            - experiment_name: name of the experiment (potential subfolder in RESTORED_DATA_PATH). If None, then save directly in RESTORED_DATA_PATH.
            - masks_filename: the name of the set of masks (without extension, ex: "my_masks"). If None, then try to use self.mask and self.masks filename.
            - mask_index: the index of the mask in the set of masks. If None, then try to use self.mask, self.masks filename and self.mask_idx.
            - use_png_data: if True, the degraded image will be loaded from PNG file (=> uint values => [0,1] clipping) otherwise from npy file (=> float values can be unclipped).
            - img_ext: extension of the images (default: "png").
            - save: if True, the restored image will be saved in the RESTORED_DATA_PATH/<experiment_name> folder.
        
        RETURNS:
            - restored_image: np.ndarray containing the restored image.
            - metrics: dict {metric_name: metric_value} containing the metrics of the inpainting.
        """

        if self.model is None or self.diffusion is None:
            raise ValueError("The model and diffusion objects must be loaded before applying inpainting.")

        # load images (and kernel if specified)        
        clean_image, degraded_image = self.load_image_data(
            degraded_dataset_name=degraded_dataset_name if degraded_dataset_name is not None else "",
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            use_png_data=use_png_data,
            img_ext=img_ext,
        )

        # load kernel if necessary (otherwise use self.kernel and self.kernel_filename)
        if masks_filename is not None:
            self.load_inpainting_mask(masks_filename, mask_index)

        if (self.masks_filename is None and self.mask is not None) or (masks_filename is not None and self.mask is None):
            raise ValueError("To load a mask, please indicate both the masks set and the mask index.")
        if self.mask is None or self.masks_filename is None:
            raise ValueError("The mask must be loaded before applying inpainting.")

        # apply DiffPIR inpainting
        restored_image, metrics = apply_DiffPIR_for_inpainting(
            config=config,
            clean_image_filename=clean_image_filename,
            degraded_image_filename=degraded_image_filename,
            masks_filename=self.masks_filename,
            mask_index=self.mask_index,
            clean_image=clean_image,
            degraded_image=degraded_image,
            mask=self.mask,
            model=self.model,
            diffusion=self.diffusion,
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

        self.logger.info(50*"-") # separate logs between different images

        return restored_image, metrics


def main():

    # setup device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()

    def demo_diffpir_deblur(device: torch.device, filename: str = None):
        """ Demonstration of DiffPIR deblurring. """

        diffpir_config = DiffPIRConfig()
        diffpir_diffuser = DiffPIRDiffuser(diffpir_config, autolog="diffpir_debluring_test", device=device)

        diffpir_diffuser.load_blur_kernel(
            kernel_filename=None,
            kernel_family="levin09",
            kernel_idx=0,
        )

        img_name = "1"

        _ = diffpir_diffuser.apply_debluring(
            config=DiffPIRDeblurConfig(),
            clean_image_filename=img_name,
            degraded_image_filename=img_name if filename is None else filename,
            degraded_dataset_name="blurred_ffhq_test20",
            # kernel_filename="gaussian_kernel_05",
            save=True,
        )
            
    def demo_diffpir_inpainting(device: torch.device, filename: str = None):
        """ Demonstration of DiffPIR inpainting. """

        diffpir_config = DiffPIRConfig()
        diffpir_diffuser = DiffPIRDiffuser(diffpir_config, autolog="diffpir_inpainting_test", device=device)

        masks_name = "random_masks"
        mask_index = 0
        diffpir_diffuser.load_mask(masks_name, mask_index)    

        img_name = "69037"

        _ = diffpir_diffuser.apply_inpainting(
            config=DiffPIRInpaintingConfig(),
            clean_image_filename=img_name,
            degraded_image_filename=img_name if filename is None else filename,
            degraded_dataset_name="masked_dataset",
            masks_filename=masks_name,
            mask_index=mask_index,
            save=True,
        )
    
    parser = argparse.ArgumentParser(description="Demonstration of DiffPIR methods.")
    parser.add_argument(
        "--debluring",
        "-d",
        action="store_true", 
        help="demonstration of DiffPIR debluring."
    )
    parser.add_argument(
        "--inpainting",
        "-i",
        action="store_true",
        help="demonstration of DiffPIR inpainting."
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
        _ = DiffPIRDiffuser(DiffPIRConfig(), device=device)
    elif args.debluring:
        demo_diffpir_deblur(device, filename=args.filename)
    elif args.inpainting:
        demo_diffpir_inpainting(device, filename=args.filename)


if __name__ == "__main__":
    main()
    # example command 1: python diffpir_diffuser.py -d -x 4
    # example command 2: python diffpir_diffuser.py -i -x 8