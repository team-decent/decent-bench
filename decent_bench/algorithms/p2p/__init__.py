from ._admm import ADMM
from ._atc import ATC, AdaptThenCombine
from ._atc_tracking import ATCT, NEXT, SONATA, ATC_Tracking
from ._atg import ATG, ADMM_Tracking, ADMM_TrackingGradient
from ._aug_dgm import ATC_DIGing, AugDGM
from ._dgd import DGD
from ._dinno import DiNNO
from ._dlm import DLM, DecentralizedLinearizedADMM
from ._ed import ED, ExactDiffusion
from ._extra import EXTRA
from ._gt_saga import GT_SAGA
from ._gt_sarah import GT_SARAH
from ._gt_vr import GT_VR
from ._kgt import KGT
from ._led import LED
from ._lt_admm import LT_ADMM
from ._lt_admm_vr import LT_ADMM_VR
from ._nids import NIDS
from ._p2p_algorithm import P2PAlgorithm
from ._prox_skip import ProxSkip
from ._simple_gt import SimpleGradientTracking, SimpleGT
from ._wang_elia import WangElia

__all__ = [
    "ADMM",
    "ATC",
    "ATCT",
    "ATG",
    "DGD",
    "DLM",
    "ED",
    "EXTRA",
    "GT_SAGA",
    "GT_SARAH",
    "GT_VR",
    "KGT",
    "LED",
    "LT_ADMM",
    "LT_ADMM_VR",
    "NEXT",
    "NIDS",
    "SONATA",
    "ADMM_Tracking",
    "ADMM_TrackingGradient",
    "ATC_DIGing",
    "ATC_Tracking",
    "AdaptThenCombine",
    "AugDGM",
    "DecentralizedLinearizedADMM",
    "DiNNO",
    "ExactDiffusion",
    "P2PAlgorithm",
    "ProxSkip",
    "SimpleGT",
    "SimpleGradientTracking",
    "WangElia",
]
