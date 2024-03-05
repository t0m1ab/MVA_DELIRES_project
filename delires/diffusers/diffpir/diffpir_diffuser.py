import os
from pathlib import Path
import numpy as np
import torch

from delires.data import load_image, load_downsample_kernel, load_blur_kernel
from delires.diffusers.diffuser import Diffuser
from delires.diffusers.diffpir.configs import DiffPIRConfig, DiffPIRDeblurConfig
from delires.diffusers.diffpir.utils import utils_image
from delires.diffusers.diffpir.utils import utils_model
from delires.diffusers.diffpir.guided_diffusion.unet import UNetModel
from delires.diffusers.diffpir.guided_diffusion.respace import SpacedDiffusion
from delires.diffusers.diffpir.my_ddpir_deblur import apply_DiffPIR_for_deblurring

from delires.diffusers.diffpir.guided_diffusion.script_util import (
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
)


class DiffPIRDiffuser(Diffuser):

    def __init__(self, config: DiffPIRConfig, **kwargs):

        self.config = config
    
        self.model: UNetModel = None
        self.diffusion: SpacedDiffusion = None
        self.load_model(config) # store in self.model and self.diffusion
        self.device = config.device

        # SISR
        self.classical_degradation = kwargs.get("sisr_classical_degradation", False)
        self.sf: int = kwargs.get("sf", 4)

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
    
    def load_model(self, config: DiffPIRConfig) -> None:
        """ Load the model and diffusion objects from the given config. """

        model_path = os.path.join(MODELS_PATH, f"{config.model_name}.pt")
        
        if config.model_name == "diffusion_ffhq_10m":
            model_config = dict(
                model_path=model_path,
                num_channels=128,
                num_res_blocks=1,
                attention_resolutions="16",
            )
        else:
            model_config = dict(
                model_path=model_path,
                num_channels=256,
                num_res_blocks=2,
                attention_resolutions="8,16,32",
            )

        args = utils_model.create_argparser(model_config).parse_args([])
        model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
        model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

        self.model = model
        self.diffusion = diffusion

    def save_restored_image(
            self, 
            restored_image: np.ndarray, 
            restored_image_filename: str,
            img_ext: str = "png",
        ):
        utils_image.imsave(restored_image, os.path.join(RESTORED_DATA_PATH, f"{restored_image_filename}.{img_ext}"))
        
    def apply_debluring(
            self,
            config: DiffPIRDeblurConfig,
            clean_image_filename: str,
            degraded_image_filename: str,
            kernel_filename: str = None,
            img_ext: str = "png",
        ) -> None:

        if self.model is None or self.diffusion is None:
            raise ValueError("The model and diffusion objects must be loaded before applying deblurring.")

        # load images

        clean_image_path = os.path.join(CLEAN_DATA_PATH, f"{clean_image_filename}.{img_ext}")
        degraded_image_path = os.path.join(DEGRADED_DATA_PATH, "blurred_dataset/", f"{degraded_image_filename}.{img_ext}")

        clean_image = load_image(clean_image_path)
        degraded_image = load_image(degraded_image_path)

        # load kernel if necessary (otherwise use self.kernel and self.kernel_filename)
        if kernel_filename is not None:
            self.load_blur_kernel(kernel_filename)

        if self.kernel is None or self.kernel_filename is None:
            raise ValueError("The blur kernel must be loaded before applying deblurring.")
        

        # apply DiffPIR deblurring
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
        )

        # save restored image
        self.save_restored_image(
            restored_image=restored_image,
            restored_image_filename=f"{degraded_image_filename}_{config.model_name}",
            img_ext=img_ext,
        )

        # save metrics
        print(f"NEED TO SAVE METRICS: {metrics}")
    
    def apply_sisr(self, degraded_image: np.ndarray):
        pass


def main():

    diffpir_config = DiffPIRConfig()
    diffpir_diffuser = DiffPIRDiffuser(diffpir_config)

    diffpir_diffuser.load_blur_kernel("gaussian_kernel_05")
    # diffpir_diffuser.load_blur_kernel("motion_kernel_1")

    diffpir_deblur_config = DiffPIRDeblurConfig()
    img_id = 69037
    diffpir_diffuser.apply_debluring(
        config=diffpir_deblur_config,
        clean_image_filename=str(img_id),
        degraded_image_filename=str(img_id),
        # kernel_filename="blur_kernel_1",
    )


if __name__ == "__main__":
    main()