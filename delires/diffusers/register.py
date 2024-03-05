from typing import Literal

from delires.diffusers.diffpir.diffpir_diffuser import DiffPIRDiffuser


DIFFUSER_TYPE = Literal["diffpir"]

DIFFUSERS = {
    "diffpir": DiffPIRDiffuser,
}