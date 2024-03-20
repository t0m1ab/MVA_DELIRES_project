#!/bin/bash

# download blur kernels
python delires/utils/download_matlab_kernels.py

# download diffusion networks
python delires/utils/utils.py

# create blur kernels
# python delires/data.py --kernel

# create inpainting masks
python delires/data.py --mask