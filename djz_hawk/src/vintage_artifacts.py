"""
Vintage Artifacts Generator for DJZ-Hawk
Recreates authentic 1996 DECtalk audio processing characteristics
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional

class VintageArtifactGenerator:
    """
    Generates authentic 1996 DECtalk audio artifacts and characteristics
    Recreates the distinctive sound of ISA card audio processing
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
    def apply_isa_card_characteristics(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply characteristics of 1996 ISA sound card processing
        Includes quantization, filtering, and DAC characteristics
        """
        # 1. Characteristic ISA card frequency response
        waveform = self._apply_isa_frequency_response(waveform)
        
        # 2. 16-bit DAC characteristics with some nonlinearity
        waveform = self._apply_vintage_dac(waveform)
        
        # 3. Characteristic ground loop hum (very subtle)
        waveform = self._add_power_line_hum(waveform)
        
        # 4. Typical 1996 anti-aliasing filter
        waveform = self._apply_vintage_antialiasing(waveform)
        
        return waveform
    
    def _apply_isa_frequency_response(self, waveform: np.ndarray) -> np.ndarray:
        """Apply characteristic ISA card frequency response"""
        # Typical 1996 sound card response: slight high-freq rolloff, mid boost
        
        # High frequency rolloff starting around 8kHz
        high_cutoff = 8000 / self.nyquist
        b_high, a_high = signal.butter(2, high_cutoff, btype='lowpass')
        waveform = signal.filtfilt(b_high, a_high, waveform)
        
        # Slight mid-frequency boost around 2-4kHz (typical of cheap DACs)
        mid_freq = [2000 / self.nyquist, 4000 / self.nyquist]
        b_mid, a_mid = signal.butter(1, mid_freq, btype='bandpass')
        mid_boost = signal.filtfilt(b_mid, a_mid, waveform) * 0.08
        
        return waveform + mid_boost
    
    def _apply_vintage_dac(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply characteristics of 1996-era 16-bit DAC
        Includes slight nonlinearity and quantization effects
        """
        # Normalize to 16-bit range
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            normalized = waveform / max_val * 0.95  # Leave headroom
        else:
            normalized = waveform
        
        # 16-bit quantization
        quantized = np.round(normalized * 32767) / 32767
        
        # Add subtle DAC nonlinearity (more pronounced at high levels)
        nonlinearity = 0.02 * quantized * np.abs(quantized)
        quantized += nonlinearity
        
        # Restore original scale
        if max_val > 0:
            quantized = quantized * max_val / 0.95
        
        return quantized
    
    def _add_power_line_hum(self, waveform: np.ndarray) -> np.ndarray:
        """Add very subtle 60Hz power line hum characteristic of 1996 hardware"""
        duration = len(waveform) / self.sample_rate
        t = np.linspace(0, duration, len(waveform))
        
        # 60Hz hum with slight 120Hz harmonic
        hum_60 = 0.0008 * np.sin(2 * np.pi * 60 * t)
        hum_120 = 0.0003 * np.sin(2 * np.pi * 120 * t)
        
        return waveform + hum_60 + hum_120
    
    def _apply_vintage_antialiasing(self, waveform: np.ndarray) -> np.ndarray:
        """Apply 1996-era anti-aliasing filter characteristics"""
        # Typical anti-aliasing: steep but not perfect rolloff at Nyquist
        cutoff = 0.85  # 85% of Nyquist frequency
        b, a = signal.butter(4, cutoff, btype='lowpass')
        return signal.filtfilt(b, a, waveform)
    
    def add_dectalk_concatenation_artifacts(self, waveform: np.ndarray, 
                                          phoneme_boundaries: List[int]) -> np.ndarray:
        """
        Add characteristic DECtalk concatenation artifacts
        These are the "clicks" and discontinuities that made DECtalk distinctive
        """
        artifact_waveform = waveform.copy()
        
        for boundary in phoneme_boundaries:
            if 0 < boundary < len(artifact_waveform) - 1:
                # Add characteristic click/pop at boundary
                click_intensity = np.random.uniform(0.05, 0.15)
                click_duration = np.random.randint(1, 4)  # 1-4 samples
                
                # Create click as sharp transient
                click = click_intensity * np.random.choice([-1, 1])
                
                # Apply click at boundary
                for i in range(click_duration):
                    if boundary + i < len(artifact_waveform):
                        artifact_waveform[boundary + i] += click * (1 - i/click_duration)
        
        return artifact_waveform
    
    def add_characteristic_beeps(self, waveform: np.ndarray, 
                               sentence_boundaries: List[int]) -> np.ndarray:
        """
        Add characteristic electronic beeps at sentence boundaries
        This was a distinctive feature of DECtalk Access32
        """
        beep_waveform = waveform.copy()
        
        for boundary in sentence_boundaries:
            if boundary < len(beep_waveform) - int(0.1 * self.sample_rate):
                # Generate subtle electronic beep
                beep_freq = np.random.uniform(780, 820)  # Around 800Hz
                beep_duration = 0.025  # 25ms
                beep_samples = int(beep_duration * self.sample_rate)
                
                t = np.linspace(0, beep_duration, beep_samples)
                beep = 0.03 * np.sin(2 * np.pi * beep_freq * t)
                
                # Apply envelope to beep
                envelope = np.exp(-t * 50)  # Fast decay
                beep *= envelope
                
                # Add beep to silence after sentence
                start_pos = boundary + int(0.05 * self.sample_rate)  # 50ms after boundary
                if start_pos + beep_samples < len(beep_waveform):
                    beep_waveform[start_pos:start_pos + beep_samples] += beep
        
        return beep_waveform
    
    def apply_dectalk_eq_characteristics(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply characteristic DECtalk EQ curve
        Emphasizes mid-frequencies for intelligibility
        """
        # DECtalk characteristic frequency response
        # Boost around 1-3kHz for consonant clarity
        consonant_freq = [1000 / self.nyquist, 3000 / self.nyquist]
        b_cons, a_cons = signal.butter(2, consonant_freq, btype='bandpass')
        consonant_boost = signal.filtfilt(b_cons, a_cons, waveform) * 0.12
        
        # Slight reduction in very low frequencies (< 200Hz)
        low_cutoff = 200 / self.nyquist
        b_low, a_low = signal.butter(1, low_cutoff, btype='highpass')
        waveform = signal.filtfilt(b_low, a_low, waveform)
        
        return waveform + consonant_boost

class DECtalkAlveolarProcessor:
    """
    Handles the characteristic alveolar->dental stop assimilation
    This was a known "feature" of DECtalk 4.2CD
    """
    
    def __init__(self):
        self.alveolar_phonemes = ['t', 'd', 'n', 'l', 's', 'z']
        self.dental_substitutions = {
            't': 'th_unvoiced',
            'd': 'th_voiced', 
            'n': 'n_dental',
            'l': 'l_dental',
            's': 's_dental',
            'z': 'z_dental'
        }
    
    def process_phoneme_sequence(self, phonemes: List[str]) -> List[str]:
        """
        Apply characteristic alveolar->dental assimilation
        Occurs in specific phonetic contexts
        """
        processed = phonemes.copy()
        
        for i, phoneme in enumerate(phonemes):
            if phoneme in self.alveolar_phonemes:
                # Check context for assimilation
                if self._should_assimilate(phonemes, i):
                    if phoneme in self.dental_substitutions:
                        processed[i] = self.dental_substitutions[phoneme]
        
        return processed
    
    def _should_assimilate(self, phonemes: List[str], position: int) -> bool:
        """
        Determine if alveolar should assimilate to dental
        Based on DECtalk's specific patterns
        """
        # DECtalk tended to do this in certain word-final positions
        # and before certain vowels
        if position == len(phonemes) - 1:  # Word final
            return np.random.random() < 0.3  # 30% chance
        
        # Before front vowels
        if position < len(phonemes) - 1:
            next_phoneme = phonemes[position + 1]
            if next_phoneme in ['i', 'e', 'ae']:
                return np.random.random() < 0.25  # 25% chance
        
        return False

class DECtalkMetallicResonator:
    """
    Generates the characteristic DECtalk metallic resonance
    This was a distinctive feature of the hardware synthesis
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        
    def add_metallic_resonance(self, waveform: np.ndarray, intensity: float = 0.4) -> np.ndarray:
        """Add characteristic DECtalk metallic sound"""
        # Create resonant filter at ~3.2kHz (characteristic DECtalk frequency)
        resonant_freq = 3200  # Hz
        q_factor = 8.0
        
        # Design resonant filter
        normalized_freq = resonant_freq / self.nyquist
        b, a = signal.iirfilter(2, normalized_freq, btype='bandpass', 
                              analog=False, ftype='butter')
        
        # Apply with characteristic intensity
        metallic_component = signal.filtfilt(b, a, waveform) * (intensity * 0.15)
        
        return waveform + metallic_component
    
    def add_klatt_resonance(self, waveform: np.ndarray) -> np.ndarray:
        """Apply Dennis Klatt's characteristic vocal tract resonance"""
        # Characteristic resonance around 1.8kHz
        resonant_freq = 1800
        normalized_freq = resonant_freq / self.nyquist
        b, a = signal.iirfilter(2, normalized_freq, btype='bandpass')
        resonance = signal.filtfilt(b, a, waveform) * 0.12
        return waveform + resonance
