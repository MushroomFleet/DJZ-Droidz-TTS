"""
DJZ-Hawk rev0 - DECtalk 4.2CD (1996) Speech Synthesis Recreation
A faithful recreation of Digital Equipment Corporation's DECtalk 4.2CD speech synthesis system
"""

__version__ = "0.1.0"
__author__ = "DJZ-Hawk Development Team"
__description__ = "DECtalk 4.2CD (1996) Speech Synthesis Recreation"

from .text_processor import DECtalk96TextProcessor
from .diphone_synthesizer import DECtalkDiphoneSynthesizer
from .voice_models import DECtalkVoiceManager
from .audio_output import AudioOutput

__all__ = [
    'DECtalk96TextProcessor',
    'DECtalkDiphoneSynthesizer', 
    'DECtalkVoiceManager',
    'AudioOutput'
]
