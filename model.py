"""Checkpoints saved before the package layout pickle `model.AnuLMConfig`.

This alias lets them load. New code imports anulm.model.
"""
import sys

from anulm import model as _model

sys.modules[__name__] = _model
