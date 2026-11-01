"""Compatibility module for old checkpoints. New code uses anulm.model."""
import sys
from anulm import model as _implementation
sys.modules[__name__] = _implementation
