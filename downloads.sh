#!/bin/bash

cd delires/utils/

# download blur kernels
python blur_kernels.py

# download diffusion networks
python utils.py

cd ../

# create blur kernels
# python data.py --kernel

# create inpainting masks
python data.py --mask