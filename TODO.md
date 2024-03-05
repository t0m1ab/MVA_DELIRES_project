# Methods

For each method in [DPS, PiGDM, DiffPIR],

For each degradation/application in [Gaussian blur, Motion blur, (Inpainting, SR)],

we need to do the following:

Implement a function `apply_METHODNAME` that takes as inputs:

* the degraded image $y$
* various parameters like the blur kernel or the number of steps...

and returns:

* the restored image $\hat{x}$
* all the images produced at intermediary steps $x_t$

# Metrics

Implement a function `compute_metrics` that takes as inputs:

* the degraded image $y$
* the original image $x$
* the restored image $\hat{x}$
* various parameters like the degradation method, the blur kernel...

and returns all the required metrics. 


At short term:
* Remove duplicates between `delires/diffusers/diffpir` and `delires`
* create the apply_ methods: both of us
* Implement metrics
* Finish pipeline to save results
* Run experiments for DiffPIR
* Do DPS/PiGDM...

