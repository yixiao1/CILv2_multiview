"""
    A simple factory module that returns instances of possible modules 

"""

from .models import CILv2_multiview_attention


def Models(configuration, rank):

    # Baseline end-to-end behavior cloning model, with TFM in multi-view feature space
    return CILv2_multiview_attention(configuration, rank)
