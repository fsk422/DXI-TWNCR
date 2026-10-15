"""Baseline detectors re-exposed under a unified import surface.

IncLOF and HST are aliased to their ``*_fixed`` implementations to bypass
known defects of the original wrappers (ratio-score / constant-score bugs).
"""
from .incremental_lof_fixed import IncrementalLOFFixed as IncrementalLOF
from .half_space_trees_fixed import HalfSpaceTreesFixed as HalfSpaceTrees
from .rrcf_wrapper import RRCFWrapper
from .xstream_detector import XStreamDetector
from .loda_wrapper import LODAIncremental
from .hbos_wrapper import HBOSIncremental
from .ecod_wrapper import ECODIncremental
from .bwoaif_wrapper import BWOAIFDIFFI
from .eilof_wrapper import EILOFDetector
from .siforest_wrapper import SiForestDetector

__all__ = [
    "IncrementalLOF", "HalfSpaceTrees", "RRCFWrapper", "XStreamDetector",
    "LODAIncremental", "HBOSIncremental", "ECODIncremental",
    "BWOAIFDIFFI", "EILOFDetector", "SiForestDetector",
]
