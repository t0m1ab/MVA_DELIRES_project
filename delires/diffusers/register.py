from typing import Literal

from delires.diffusers.diffpir.diffpir_diffuser import DiffPIRDiffuser
from delires.diffusers.dps.dps_diffuser import DPSDiffuser


DIFFUSER_TYPE = Literal["diffpir"]

DIFFUSERS = {
    "diffpir": DiffPIRDiffuser,
    "dps": DPSDiffuser
}