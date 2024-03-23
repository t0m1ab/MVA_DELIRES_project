from typing import Literal, Union

from delires.methods.dps.dps_diffuser import DPSDiffuser
from delires.methods.dps.dps_configs import DPSConfig, DPSDeblurConfig, DPSInpaintingConfig
from delires.methods.pigdm.pigdm_diffuser import PiGDMDiffuser
from delires.methods.pigdm.pigdm_configs import PiGDMConfig, PiGDMDeblurConfig, PiGDMInpaintingConfig
from delires.methods.diffpir.diffpir_diffuser import DiffPIRDiffuser
from delires.methods.diffpir.diffpir_configs import DiffPIRConfig, DiffPIRDeblurConfig, DiffPIRInpaintingConfig


DIFFUSERS = {
    "dps": DPSDiffuser,
    "pigdm": PiGDMDiffuser,
    "diffpir": DiffPIRDiffuser,
}

DIFFUSER_CONFIG = {
    "diffpir": DiffPIRConfig(),
    "dps": DPSConfig(),
    "pigdm": PiGDMConfig(),
}

TASKS = [
    "deblur", 
    "inpaint",
]

TASK_CONFIG = Union[
    DPSDeblurConfig,
    DPSInpaintingConfig,
    PiGDMDeblurConfig,
    PiGDMInpaintingConfig,
    DiffPIRDeblurConfig,
    DiffPIRInpaintingConfig,
]