#!/bin/bash

cd models/diffPIR
bash download.sh
# rename "ffhq_10m" to "diffusion_ffhq_m" for code consistency
mv model_zoo/ffhq_10m model_zoo/diffusion_ffhq_m