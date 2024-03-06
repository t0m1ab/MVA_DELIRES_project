from typing import Literal

from delires.methods.diffpir.diffpir_diffuser import DiffPIRDiffuser
from delires.methods.dps.dps_diffuser import DPSDiffuser


DIFFUSER_TYPE = Literal["diffpir"]

DIFFUSERS = {
    "diffpir": DiffPIRDiffuser,
    "dps": DPSDiffuser
}