#!/bin/bash

# download matlab blur kernels
python utils/matlab_kernels.py

# download diffusion networks
python utils/utils.py

# create blur kernels
# python utils/operators.py --kernel -x 20 -s 42 -p

# create inpainting masks
python utils/operators.py --mask -x 20 -s 42 -p

# create blurred dataset
python data.py --kernel kernels_12 -n 15 -s 42

# create masked dataset
python data.py --mask box_masks -n 15 -s 42