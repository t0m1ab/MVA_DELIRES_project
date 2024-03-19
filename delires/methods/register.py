from typing import Literal, Union

from delires.methods.diffpir.diffpir_diffuser import DiffPIRDiffuser
from delires.methods.diffpir.diffpir_configs import DiffPIRConfig, DiffPIRDeblurConfig, DiffPIRInpaintingConfig
from delires.methods.dps.dps_diffuser import DPSDiffuser
from delires.methods.dps.dps_configs import DPSConfig, DPSDeblurConfig
from delires.methods.pigdm.pigdm_diffuser import PiGDMDiffuser
from delires.methods.pigdm.pigdm_configs import PiGDMConfig, PiGDMDeblurConfig


DIFFUSER_TYPE = Literal["diffpir", "dps", "pigdm"]

DIFFUSERS = {
    "diffpir": DiffPIRDiffuser,
    "dps": DPSDiffuser,
    "pigdm": PiGDMDiffuser,
}

DIFFUSER_CONFIG = Union[
    DiffPIRConfig,
    DPSConfig,
    PiGDMConfig,
]

TASK_CONFIG = Union[
    DiffPIRDeblurConfig,
    DiffPIRInpaintingConfig,
    DPSDeblurConfig,
    PiGDMDeblurConfig,
]