# At short term:
* Remove duplicates between `delires/diffusers/diffpir` and `delires`
* create the apply_ methods: both of us
* Implement metrics
* Finish pipeline to save results
* Run experiments for DiffPIR
* Do DPS/PiGDM...

# To check:
* use different kernels for each image in a blurred_dataset and encode mapping in JSON dataset infos
* adapt main (invert loops to alternate between methods during full experiment ?)
* add option to save intermediate generations in each method (desactivate when running experiment to avoid overload)
