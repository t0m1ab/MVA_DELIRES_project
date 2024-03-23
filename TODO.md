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

# Interestering images for qualitative comparison and comments

## Deblurring

* 69010: diffpir impressing very good - pigdm has a lot of artefacts - dps far for original
* 69025/69037: pigdm struggles with regions that look like noise (the short beard of the guy / short hair of the girl)
* 69042/69043/69051/69062: pigdm struggles with regions that look like noise (blurred background)
* 69028: dps struggles with hair and textures that are not part of human face (usually on clothes or in the background)
* 69060: dps considerably simplified the background (no tiles on the wall anymore)
* 69069/69072: sometimes diffpir seems noisier than pigdm...
* 69076: dps seems to struggle with unusual objects on the image (like the microphone)
* 69093: dps doesn't like writings apparently...
* 69094/69098: pigdm seems to suffer numerical instability in some cases leading to catastrophic results

## Inpainting

In general, pigdm fails to reconstruct any good looking area at the position of the mask and diffpir gives noisy areas everywhere but at the mask's position.
* 69019/69046/69048/69099: good perf of dps and diffpir (even if noisy areas with diffpir)
* 69037: diffpir totally obliterates the painting on the face of the girl
* 69042/69052/69072/69092: diffpir doesn't like glasses apparently
* 69069/69071: dps likes glasses apparently
* 69056: diffpir gives slighlty better results than dps