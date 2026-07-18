"""
Attack module for red-team testing of skills-based agent security.
This module is designed for authorized security research purposes only.
"""

from .crypto_utils import CryptoManager
from .target import BackdoorTarget
from .generate_backdoor_skills import SkillBackdoorGenerator

__all__ = ['CryptoManager', 'BackdoorTarget', 'SkillBackdoorGenerator']
