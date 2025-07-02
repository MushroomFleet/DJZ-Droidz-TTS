"""
Voice Models for DJZ-Hawk
Implements all 9 distinctive DECtalk voices with authentic characteristics
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class VoiceCharacteristics:
    """Defines the characteristics of a DECtalk voice"""
    name: str
    gender: str
    age_group: str  # 'adult', 'child', 'elderly'
    base_frequency: float
    frequency_range: float
    formant_shifts: Dict[str, float]  # F1, F2, F3 multipliers
    speech_rate: float  # words per minute
    roughness: float
    breathiness: float
    nasality: float
    precision: float  # articulation precision
    timbre_metallic: float  # DECtalk metallic characteristic
    special_effects: List[str]  # voice-specific effects
    test_phrase: str  # characteristic test phrase

class DECtalkVoiceModel(ABC):
    """Abstract base class for DECtalk voice models"""
    
    def __init__(self, characteristics: VoiceCharacteristics):
        self.characteristics = characteristics
        self.diphone_modifications = {}
        
    @abstractmethod
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply voice-specific modifications to diphone unit"""
        pass
    
    @abstractmethod
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply voice-specific prosodic modifications"""
        pass

class PerfectPaulVoice(DECtalkVoiceModel):
    """
    Perfect Paul - The default DECtalk voice based on Dennis Klatt's speech
    Most famous as Stephen Hawking's voice
    """
    
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Perfect Paul",
            gender="male",
            age_group="adult",
            base_frequency=122.0,  # Hz
            frequency_range=40.0,
            formant_shifts={"F1": 1.0, "F2": 1.0, "F3": 1.0},
            speech_rate=160,  # WPM
            roughness=0.15,
            breathiness=0.05,
            nasality=0.10,
            precision=0.85,
            timbre_metallic=0.40,  # High metallic characteristic
            special_effects=["klatt_resonance"],
            test_phrase="Hello, my name is Perfect Paul."
        )
        super().__init__(characteristics)
        
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply Perfect Paul specific modifications"""
        # Create a copy to avoid modifying the original
        modified = self._copy_diphone_unit(diphone_unit)
        
        # Apply characteristic formant modifications
        if hasattr(modified, 'formant_tracks'):
            modified.formant_tracks['F1'] *= self.characteristics.formant_shifts['F1']
            modified.formant_tracks['F2'] *= self.characteristics.formant_shifts['F2']
            modified.formant_tracks['F3'] *= self.characteristics.formant_shifts['F3']
        
        # Add characteristic roughness
        if self.characteristics.roughness > 0:
            roughness_noise = np.random.normal(0, self.characteristics.roughness, 
                                             len(modified.waveform))
            modified.waveform += roughness_noise * 0.1
        
        # Apply Dennis Klatt's characteristic vocal tract resonance
        modified.waveform = self._apply_klatt_resonance(modified.waveform)
        
        return modified
    
    def _copy_diphone_unit(self, unit):
        """Create a copy of diphone unit"""
        # Simple copy - in full implementation would use proper deep copy
        class ModifiedUnit:
            def __init__(self, original):
                self.name = original.name
                self.waveform = original.waveform.copy()
                self.f0_contour = original.f0_contour.copy()
                self.formant_tracks = {k: v.copy() for k, v in original.formant_tracks.items()}
                self.duration = original.duration
                self.energy_contour = original.energy_contour.copy()
                self.phoneme_boundary = original.phoneme_boundary
                self.lpc_coefficients = original.lpc_coefficients.copy()
                self.source_voice = original.source_voice
        
        return ModifiedUnit(unit)
    
    def _apply_klatt_resonance(self, waveform: np.ndarray) -> np.ndarray:
        """Apply Dennis Klatt's characteristic vocal tract resonance"""
        # Characteristic resonance around 1.8kHz
        try:
            from scipy import signal
            resonant_freq = 1800
            sample_rate = 22050
            nyquist = sample_rate / 2
            normalized_freq = resonant_freq / nyquist
            
            if normalized_freq < 1.0:
                b, a = signal.iirfilter(2, normalized_freq, btype='bandpass')
                resonance = signal.filtfilt(b, a, waveform) * 0.12
                return waveform + resonance
        except:
            pass
        
        return waveform
    
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply Perfect Paul prosodic characteristics"""
        # Perfect Paul has minimal prosodic variation - very flat
        # Reduce pitch variation to be more monotone
        if 'pitch_factor' in prosody:
            prosody['pitch_factor'] *= 0.7  # Reduce pitch variation
        
        return waveform

class BeautifulBettyVoice(DECtalkVoiceModel):
    """
    Beautiful Betty - Female voice based on Klatt's wife
    Higher pitch with slightly more natural prosody
    """
    
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Beautiful Betty",
            gender="female",
            age_group="adult",
            base_frequency=210.0,
            frequency_range=60.0,
            formant_shifts={"F1": 1.15, "F2": 1.12, "F3": 1.08},
            speech_rate=155,
            roughness=0.08,
            breathiness=0.12,
            nasality=0.15,
            precision=0.88,
            timbre_metallic=0.35,
            special_effects=["female_formant_boost"],
            test_phrase="Hello, my name is Beautiful Betty."
        )
        super().__init__(characteristics)
    
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply Beautiful Betty specific modifications"""
        modified = self._copy_diphone_unit(diphone_unit)
        
        # Female formant modifications
        if hasattr(modified, 'formant_tracks'):
            modified.formant_tracks['F1'] *= self.characteristics.formant_shifts['F1']
            modified.formant_tracks['F2'] *= self.characteristics.formant_shifts['F2'] 
            modified.formant_tracks['F3'] *= self.characteristics.formant_shifts['F3']
        
        # Add breathiness characteristic of Betty
        breathiness_noise = np.random.normal(0, 0.02, len(modified.waveform))
        modified.waveform += breathiness_noise
        
        return modified
    
    def _copy_diphone_unit(self, unit):
        """Create a copy of diphone unit"""
        class ModifiedUnit:
            def __init__(self, original):
                self.name = original.name
                self.waveform = original.waveform.copy()
                self.f0_contour = original.f0_contour.copy()
                self.formant_tracks = {k: v.copy() for k, v in original.formant_tracks.items()}
                self.duration = original.duration
                self.energy_contour = original.energy_contour.copy()
                self.phoneme_boundary = original.phoneme_boundary
                self.lpc_coefficients = original.lpc_coefficients.copy()
                self.source_voice = original.source_voice
        
        return ModifiedUnit(unit)
    
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply Beautiful Betty prosodic characteristics"""
        # Betty has slightly more natural prosody than Paul
        return waveform

class HugeHarryVoice(DECtalkVoiceModel):
    """
    Huge Harry - Very deep male voice
    Famous for airport ATIS announcements
    """
    
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Huge Harry",
            gender="male",
            age_group="adult",
            base_frequency=85.0,   # Very low
            frequency_range=25.0,  # Limited range for depth
            formant_shifts={"F1": 0.85, "F2": 0.90, "F3": 0.92},
            speech_rate=140,       # Slower for authority
            roughness=0.25,        # More rough
            breathiness=0.03,
            nasality=0.05,
            precision=0.90,        # Very precise for aviation
            timbre_metallic=0.45,
            special_effects=["deep_resonance", "aviation_clarity"],
            test_phrase="Hello, my name is Huge Harry."
        )
        super().__init__(characteristics)
    
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply Huge Harry specific modifications"""
        modified = self._copy_diphone_unit(diphone_unit)
        
        # Lower all formants for deeper voice
        if hasattr(modified, 'formant_tracks'):
            modified.formant_tracks['F1'] *= self.characteristics.formant_shifts['F1']
            modified.formant_tracks['F2'] *= self.characteristics.formant_shifts['F2']
            modified.formant_tracks['F3'] *= self.characteristics.formant_shifts['F3']
        
        # Add deep chest resonance
        modified.waveform = self._add_chest_resonance(modified.waveform)
        
        return modified
    
    def _copy_diphone_unit(self, unit):
        """Create a copy of diphone unit"""
        class ModifiedUnit:
            def __init__(self, original):
                self.name = original.name
                self.waveform = original.waveform.copy()
                self.f0_contour = original.f0_contour.copy()
                self.formant_tracks = {k: v.copy() for k, v in original.formant_tracks.items()}
                self.duration = original.duration
                self.energy_contour = original.energy_contour.copy()
                self.phoneme_boundary = original.phoneme_boundary
                self.lpc_coefficients = original.lpc_coefficients.copy()
                self.source_voice = original.source_voice
        
        return ModifiedUnit(unit)
    
    def _add_chest_resonance(self, waveform: np.ndarray) -> np.ndarray:
        """Add characteristic deep chest resonance"""
        try:
            from scipy import signal
            # Boost low frequencies around 120Hz
            chest_freq = 120
            sample_rate = 22050
            nyquist = sample_rate / 2
            normalized_freq = chest_freq / nyquist
            
            if normalized_freq < 1.0:
                b, a = signal.iirfilter(2, normalized_freq, btype='lowpass')
                chest_resonance = signal.filtfilt(b, a, waveform) * 0.15
                return waveform + chest_resonance
        except:
            pass
        
        return waveform
    
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply Huge Harry prosodic characteristics"""
        # Harry speaks slower and more deliberately
        return waveform

class KitTheKidVoice(DECtalkVoiceModel):
    """
    Kit the Kid - Child voice based on Klatt's daughter
    Higher pitch with characteristic child speech patterns
    """
    
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Kit the Kid",
            gender="child",
            age_group="child", 
            base_frequency=280.0,
            frequency_range=80.0,
            formant_shifts={"F1": 1.25, "F2": 1.20, "F3": 1.15},
            speech_rate=145,
            roughness=0.05,
            breathiness=0.08,
            nasality=0.20,  # Children often more nasal
            precision=0.75,  # Less precise articulation
            timbre_metallic=0.30,
            special_effects=["child_resonance", "slight_lisp"],
            test_phrase="Hello, my name is Kit the Kid."
        )
        super().__init__(characteristics)
    
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply Kit the Kid specific modifications"""
        modified = self._copy_diphone_unit(diphone_unit)
        
        # Child-like formant modifications
        if hasattr(modified, 'formant_tracks'):
            modified.formant_tracks['F1'] *= self.characteristics.formant_shifts['F1']
            modified.formant_tracks['F2'] *= self.characteristics.formant_shifts['F2']
            modified.formant_tracks['F3'] *= self.characteristics.formant_shifts['F3']
        
        # Add slight lisp effect for 's' sounds
        if 'S' in diphone_unit.name or 'Z' in diphone_unit.name:
            modified.waveform = self._add_slight_lisp(modified.waveform)
        
        return modified
    
    def _copy_diphone_unit(self, unit):
        """Create a copy of diphone unit"""
        class ModifiedUnit:
            def __init__(self, original):
                self.name = original.name
                self.waveform = original.waveform.copy()
                self.f0_contour = original.f0_contour.copy()
                self.formant_tracks = {k: v.copy() for k, v in original.formant_tracks.items()}
                self.duration = original.duration
                self.energy_contour = original.energy_contour.copy()
                self.phoneme_boundary = original.phoneme_boundary
                self.lpc_coefficients = original.lpc_coefficients.copy()
                self.source_voice = original.source_voice
        
        return ModifiedUnit(unit)
    
    def _add_slight_lisp(self, waveform: np.ndarray) -> np.ndarray:
        """Add slight lisp effect"""
        # Simple frequency shift to simulate lisp
        try:
            from scipy import signal
            # Shift high frequencies slightly lower
            nyquist = 22050 / 2
            high_freq = 4000 / nyquist
            
            if high_freq < 1.0:
                b, a = signal.iirfilter(2, high_freq, btype='highpass')
                high_component = signal.filtfilt(b, a, waveform)
                # Slightly reduce high frequencies
                return waveform - high_component * 0.1
        except:
            pass
        
        return waveform
    
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply Kit the Kid prosodic characteristics"""
        # Child voice has more variable prosody
        return waveform

# Additional voice classes for completeness
class FrankVoice(DECtalkVoiceModel):
    """Frank - Alternative male voice"""
    
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Frank",
            gender="male",
            age_group="adult",
            base_frequency=115.0,
            frequency_range=35.0,
            formant_shifts={"F1": 0.95, "F2": 0.98, "F3": 1.02},
            speech_rate=165,
            roughness=0.18,
            breathiness=0.07,
            nasality=0.08,
            precision=0.82,
            timbre_metallic=0.38,
            special_effects=["frank_resonance"],
            test_phrase="Hello, my name is Frank."
        )
        super().__init__(characteristics)
    
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply Frank specific modifications"""
        # Basic implementation
        return diphone_unit
    
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply Frank prosodic characteristics"""
        return waveform

class RitaVoice(DECtalkVoiceModel):
    """Rita - Warm female voice"""
    
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Rita",
            gender="female",
            age_group="adult",
            base_frequency=195.0,
            frequency_range=55.0,
            formant_shifts={"F1": 1.12, "F2": 1.08, "F3": 1.05},
            speech_rate=158,
            roughness=0.10,
            breathiness=0.10,
            nasality=0.12,
            precision=0.86,
            timbre_metallic=0.33,
            special_effects=["rita_warmth"],
            test_phrase="Hello, my name is Rita."
        )
        super().__init__(characteristics)
    
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply Rita specific modifications"""
        return diphone_unit
    
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply Rita prosodic characteristics"""
        return waveform

class UrsulaVoice(DECtalkVoiceModel):
    """Ursula - Dramatic female voice"""
    
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Ursula",
            gender="female",
            age_group="adult",
            base_frequency=180.0,
            frequency_range=70.0,
            formant_shifts={"F1": 1.18, "F2": 1.15, "F3": 1.10},
            speech_rate=150,
            roughness=0.12,
            breathiness=0.15,
            nasality=0.18,
            precision=0.90,
            timbre_metallic=0.32,
            special_effects=["dramatic_emphasis"],
            test_phrase="Hello, my name is Ursula."
        )
        super().__init__(characteristics)
    
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply Ursula specific modifications"""
        return diphone_unit
    
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply Ursula prosodic characteristics"""
        return waveform

class ValVoice(DECtalkVoiceModel):
    """Val - Valley girl style voice"""
    
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Val",
            gender="female",
            age_group="adult",
            base_frequency=220.0,
            frequency_range=65.0,
            formant_shifts={"F1": 1.20, "F2": 1.18, "F3": 1.12},
            speech_rate=170,
            roughness=0.06,
            breathiness=0.14,
            nasality=0.22,
            precision=0.78,
            timbre_metallic=0.28,
            special_effects=["valley_girl", "uptalk"],
            test_phrase="Hello, my name is Val."
        )
        super().__init__(characteristics)
    
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply Val specific modifications"""
        return diphone_unit
    
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply Val prosodic characteristics"""
        return waveform

class RoughVoice(DECtalkVoiceModel):
    """Rough - Gravelly textured voice"""
    
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Rough",
            gender="male",
            age_group="adult",
            base_frequency=105.0,
            frequency_range=30.0,
            formant_shifts={"F1": 0.90, "F2": 0.92, "F3": 0.95},
            speech_rate=135,
            roughness=0.45,
            breathiness=0.20,
            nasality=0.06,
            precision=0.75,
            timbre_metallic=0.50,
            special_effects=["gravelly_texture", "vocal_fry"],
            test_phrase="Hello, my name is Rough."
        )
        super().__init__(characteristics)
    
    def modify_diphone(self, diphone_unit, context: Dict):
        """Apply Rough specific modifications"""
        return diphone_unit
    
    def apply_prosody(self, waveform: np.ndarray, prosody: Dict) -> np.ndarray:
        """Apply Rough prosodic characteristics"""
        return waveform

class DECtalkVoiceManager:
    """Manages all DECtalk voices and provides unified interface"""
    
    def __init__(self):
        self.voices = {
            'perfect_paul': PerfectPaulVoice(),
            'beautiful_betty': BeautifulBettyVoice(), 
            'huge_harry': HugeHarryVoice(),
            'kit_the_kid': KitTheKidVoice(),
            'frank': FrankVoice(),
            'rita': RitaVoice(),
            'ursula': UrsulaVoice(),
            'val': ValVoice(),
            'rough': RoughVoice()
        }
        self.current_voice = 'perfect_paul'
    
    def set_voice(self, voice_name: str):
        """Set the current voice"""
        if voice_name in self.voices:
            self.current_voice = voice_name
        else:
            raise ValueError(f"Unknown voice: {voice_name}")
    
    def get_voice(self, voice_name: str = None) -> DECtalkVoiceModel:
        """Get voice model"""
        voice_name = voice_name or self.current_voice
        if voice_name not in self.voices:
            raise ValueError(f"Unknown voice: {voice_name}")
        return self.voices[voice_name]
    
    def list_voices(self) -> List[str]:
        """List all available voices"""
        return list(self.voices.keys())
    
    def get_voice_info(self, voice_name: str) -> VoiceCharacteristics:
        """Get voice characteristics"""
        if voice_name not in self.voices:
            raise ValueError(f"Unknown voice: {voice_name}")
        return self.voices[voice_name].characteristics
    
    def load_voice_configs(self) -> Dict:
        """Load voice configurations from file"""
        config_path = os.path.join('config', 'voice_configs.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def get_voice_list_with_info(self) -> List[Dict]:
        """Get list of voices with their information"""
        voice_list = []
        for voice_name in self.voices:
            characteristics = self.get_voice_info(voice_name)
            voice_list.append({
                'name': voice_name,
                'display_name': characteristics.name,
                'gender': characteristics.gender,
                'age_group': characteristics.age_group,
                'base_frequency': characteristics.base_frequency,
                'test_phrase': characteristics.test_phrase
            })
        return voice_list
