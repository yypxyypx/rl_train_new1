from .base import RawSample
from . import re10k, dl3dv, scannetpp

DATASET_REGISTRY = {
    "re10k": re10k.parse,
    "dl3dv": dl3dv.parse,
    "scannetpp": scannetpp.parse,
}
