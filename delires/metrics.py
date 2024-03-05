import delires.utils.utils_image as utils_image
import numpy as np
import torch
import delires.utils.utils_image as utils_image



def compute_metrics(result_image: np.ndarray, clean_image: np.ndarray, border: int = 0, calc_LPIPS: bool = False, device: str = "cpu", loss_fn_vgg = None):
    psnr = utils_image.calculate_psnr(result_image, clean_image, border=border)
    if calc_LPIPS:
        if loss_fn_vgg is None:
            raise ValueError("loss_fn_vgg must be provided if calc_LPIPS is True.")
        clean_image_tensor = np.transpose(clean_image, (2, 0, 1))
        clean_image_tensor = torch.from_numpy(clean_image_tensor)[None,:,:,:].to(device)
        clean_image_tensor = clean_image_tensor / 255 * 2 -1
        x_0 = utils_image.tensor2uint(clean_image)
        lpips_score = loss_fn_vgg(x_0.detach()*2-1, clean_image_tensor)
        lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
        return psnr, lpips_score
    # TODO: DATA_fitting
    # TODO: variance
    # TODO: FID
    # TODO: coverage
    return psnr