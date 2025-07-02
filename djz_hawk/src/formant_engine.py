"""
Klatt Formant Synthesis Engine
Implements Dennis Klatt's formant synthesis methodology for authentic DECtalk recreation
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FormantParameters:
    """Parameters for a single formant"""
    frequency: float    # Center frequency (Hz)
    bandwidth: float    # Bandwidth (Hz)
    amplitude: float    # Amplitude (dB)

@dataclass
class PhonemeFormants:
    """Complete formant specification for a phoneme"""
    F1: FormantParameters
    F2: FormantParameters
    F3: FormantParameters
    F4: FormantParameters
    voicing: float      # 0.0 = unvoiced, 1.0 = fully voiced
    aspiration: float   # Aspiration noise level
    frication: float    # Frication noise level
    nasality: float     # Nasal coupling

class KlattFormantSynthesizer:
    """
    Implementation of Dennis Klatt's formant synthesis
    Based on the Klatt synthesizer used in DECtalk
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.nyquist = sample_rate / 2
        self.formant_data = self._load_formant_database()
        
    def _load_formant_database(self) -> Dict[str, PhonemeFormants]:
        """Load comprehensive formant database for all phonemes"""
        return {
            # Vowels - based on Peterson & Barney (1952) and Klatt modifications
            'AA': PhonemeFormants(  # father
                F1=FormantParameters(730, 60, 0),
                F2=FormantParameters(1090, 90, -6),
                F3=FormantParameters(2440, 120, -12),
                F4=FormantParameters(3400, 150, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'AE': PhonemeFormants(  # cat
                F1=FormantParameters(660, 80, 0),
                F2=FormantParameters(1720, 110, -6),
                F3=FormantParameters(2410, 140, -12),
                F4=FormantParameters(3300, 160, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'AH': PhonemeFormants(  # cut
                F1=FormantParameters(520, 70, 0),
                F2=FormantParameters(1190, 100, -6),
                F3=FormantParameters(2390, 130, -12),
                F4=FormantParameters(3400, 150, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'AO': PhonemeFormants(  # caught
                F1=FormantParameters(570, 80, 0),
                F2=FormantParameters(840, 90, -6),
                F3=FormantParameters(2250, 120, -12),
                F4=FormantParameters(3200, 140, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'AW': PhonemeFormants(  # how (diphthong - using midpoint)
                F1=FormantParameters(570, 80, 0),
                F2=FormantParameters(1200, 100, -6),
                F3=FormantParameters(2400, 130, -12),
                F4=FormantParameters(3300, 150, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'AY': PhonemeFormants(  # hide (diphthong - using midpoint)
                F1=FormantParameters(660, 80, 0),
                F2=FormantParameters(1720, 110, -6),
                F3=FormantParameters(2410, 140, -12),
                F4=FormantParameters(3300, 160, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'EH': PhonemeFormants(  # red
                F1=FormantParameters(530, 70, 0),
                F2=FormantParameters(1840, 120, -6),
                F3=FormantParameters(2480, 140, -12),
                F4=FormantParameters(3500, 160, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'ER': PhonemeFormants(  # hurt
                F1=FormantParameters(490, 70, 0),
                F2=FormantParameters(1350, 100, -6),
                F3=FormantParameters(1690, 120, -12),
                F4=FormantParameters(3300, 150, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'EY': PhonemeFormants(  # ate (diphthong - using midpoint)
                F1=FormantParameters(530, 70, 0),
                F2=FormantParameters(1840, 120, -6),
                F3=FormantParameters(2480, 140, -12),
                F4=FormantParameters(3500, 160, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'IH': PhonemeFormants(  # hit
                F1=FormantParameters(390, 60, 0),
                F2=FormantParameters(1990, 130, -6),
                F3=FormantParameters(2550, 150, -12),
                F4=FormantParameters(3600, 170, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'IY': PhonemeFormants(  # heat
                F1=FormantParameters(270, 50, 0),
                F2=FormantParameters(2290, 140, -6),
                F3=FormantParameters(3010, 160, -12),
                F4=FormantParameters(3700, 180, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'OW': PhonemeFormants(  # boat (diphthong - using midpoint)
                F1=FormantParameters(570, 80, 0),
                F2=FormantParameters(840, 90, -6),
                F3=FormantParameters(2250, 120, -12),
                F4=FormantParameters(3200, 140, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'OY': PhonemeFormants(  # boy (diphthong - using midpoint)
                F1=FormantParameters(570, 80, 0),
                F2=FormantParameters(1200, 100, -6),
                F3=FormantParameters(2400, 130, -12),
                F4=FormantParameters(3300, 150, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'UH': PhonemeFormants(  # book
                F1=FormantParameters(440, 70, 0),
                F2=FormantParameters(1020, 90, -6),
                F3=FormantParameters(2240, 120, -12),
                F4=FormantParameters(3200, 140, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'UW': PhonemeFormants(  # boot
                F1=FormantParameters(300, 60, 0),
                F2=FormantParameters(870, 80, -6),
                F3=FormantParameters(2240, 120, -12),
                F4=FormantParameters(3200, 140, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            
            # Voiced Consonants
            'B': PhonemeFormants(  # bat
                F1=FormantParameters(200, 100, -6),
                F2=FormantParameters(1000, 150, -12),
                F3=FormantParameters(2500, 200, -18),
                F4=FormantParameters(3500, 250, -24),
                voicing=0.8, aspiration=0.1, frication=0.0, nasality=0.0
            ),
            'D': PhonemeFormants(  # dog
                F1=FormantParameters(200, 100, -6),
                F2=FormantParameters(1700, 150, -12),
                F3=FormantParameters(2600, 200, -18),
                F4=FormantParameters(3600, 250, -24),
                voicing=0.8, aspiration=0.1, frication=0.0, nasality=0.0
            ),
            'G': PhonemeFormants(  # got
                F1=FormantParameters(200, 100, -6),
                F2=FormantParameters(1000, 150, -12),
                F3=FormantParameters(2500, 200, -18),
                F4=FormantParameters(3500, 250, -24),
                voicing=0.8, aspiration=0.1, frication=0.0, nasality=0.0
            ),
            'V': PhonemeFormants(  # vat
                F1=FormantParameters(200, 120, -6),
                F2=FormantParameters(1400, 180, -12),
                F3=FormantParameters(2600, 220, -18),
                F4=FormantParameters(3600, 280, -24),
                voicing=0.7, aspiration=0.0, frication=0.3, nasality=0.0
            ),
            'DH': PhonemeFormants(  # that
                F1=FormantParameters(200, 120, -6),
                F2=FormantParameters(1400, 180, -12),
                F3=FormantParameters(2600, 220, -18),
                F4=FormantParameters(3600, 280, -24),
                voicing=0.6, aspiration=0.0, frication=0.4, nasality=0.0
            ),
            'Z': PhonemeFormants(  # zoo
                F1=FormantParameters(200, 120, -6),
                F2=FormantParameters(1400, 180, -12),
                F3=FormantParameters(2600, 220, -18),
                F4=FormantParameters(3600, 280, -24),
                voicing=0.7, aspiration=0.0, frication=0.3, nasality=0.0
            ),
            'ZH': PhonemeFormants(  # measure
                F1=FormantParameters(200, 120, -6),
                F2=FormantParameters(1200, 180, -12),
                F3=FormantParameters(2400, 220, -18),
                F4=FormantParameters(3400, 280, -24),
                voicing=0.7, aspiration=0.0, frication=0.3, nasality=0.0
            ),
            'JH': PhonemeFormants(  # joy
                F1=FormantParameters(200, 120, -6),
                F2=FormantParameters(1800, 180, -12),
                F3=FormantParameters(2600, 220, -18),
                F4=FormantParameters(3600, 280, -24),
                voicing=0.8, aspiration=0.1, frication=0.2, nasality=0.0
            ),
            
            # Unvoiced Consonants
            'P': PhonemeFormants(  # pat
                F1=FormantParameters(200, 150, -12),
                F2=FormantParameters(1000, 200, -18),
                F3=FormantParameters(2500, 250, -24),
                F4=FormantParameters(3500, 300, -30),
                voicing=0.0, aspiration=0.8, frication=0.0, nasality=0.0
            ),
            'T': PhonemeFormants(  # top
                F1=FormantParameters(200, 150, -12),
                F2=FormantParameters(1700, 200, -18),
                F3=FormantParameters(2600, 250, -24),
                F4=FormantParameters(3600, 300, -30),
                voicing=0.0, aspiration=0.8, frication=0.0, nasality=0.0
            ),
            'K': PhonemeFormants(  # cat
                F1=FormantParameters(200, 150, -12),
                F2=FormantParameters(1000, 200, -18),
                F3=FormantParameters(2500, 250, -24),
                F4=FormantParameters(3500, 300, -30),
                voicing=0.0, aspiration=0.8, frication=0.0, nasality=0.0
            ),
            'F': PhonemeFormants(  # fat
                F1=FormantParameters(200, 150, -12),
                F2=FormantParameters(1400, 200, -18),
                F3=FormantParameters(2600, 250, -24),
                F4=FormantParameters(3600, 300, -30),
                voicing=0.0, aspiration=0.0, frication=0.8, nasality=0.0
            ),
            'TH': PhonemeFormants(  # think
                F1=FormantParameters(200, 150, -12),
                F2=FormantParameters(1400, 200, -18),
                F3=FormantParameters(2600, 250, -24),
                F4=FormantParameters(3600, 300, -30),
                voicing=0.0, aspiration=0.0, frication=0.8, nasality=0.0
            ),
            'S': PhonemeFormants(  # sat
                F1=FormantParameters(200, 150, -12),
                F2=FormantParameters(1400, 200, -18),
                F3=FormantParameters(2600, 250, -24),
                F4=FormantParameters(3600, 300, -30),
                voicing=0.0, aspiration=0.0, frication=0.8, nasality=0.0
            ),
            'SH': PhonemeFormants(  # shot
                F1=FormantParameters(200, 150, -12),
                F2=FormantParameters(1200, 200, -18),
                F3=FormantParameters(2400, 250, -24),
                F4=FormantParameters(3400, 300, -30),
                voicing=0.0, aspiration=0.0, frication=0.8, nasality=0.0
            ),
            'CH': PhonemeFormants(  # chat
                F1=FormantParameters(200, 150, -12),
                F2=FormantParameters(1800, 200, -18),
                F3=FormantParameters(2600, 250, -24),
                F4=FormantParameters(3600, 300, -30),
                voicing=0.0, aspiration=0.3, frication=0.5, nasality=0.0
            ),
            'HH': PhonemeFormants(  # hat
                F1=FormantParameters(200, 200, -12),
                F2=FormantParameters(1500, 250, -18),
                F3=FormantParameters(2500, 300, -24),
                F4=FormantParameters(3500, 350, -30),
                voicing=0.0, aspiration=0.9, frication=0.0, nasality=0.0
            ),
            
            # Nasals
            'M': PhonemeFormants(  # mat
                F1=FormantParameters(200, 80, -3),
                F2=FormantParameters(1000, 120, -9),
                F3=FormantParameters(2500, 160, -15),
                F4=FormantParameters(3500, 200, -21),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.8
            ),
            'N': PhonemeFormants(  # not
                F1=FormantParameters(200, 80, -3),
                F2=FormantParameters(1500, 120, -9),
                F3=FormantParameters(2600, 160, -15),
                F4=FormantParameters(3600, 200, -21),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.8
            ),
            'NG': PhonemeFormants(  # sing
                F1=FormantParameters(200, 80, -3),
                F2=FormantParameters(1200, 120, -9),
                F3=FormantParameters(2500, 160, -15),
                F4=FormantParameters(3500, 200, -21),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.8
            ),
            
            # Liquids
            'L': PhonemeFormants(  # lot
                F1=FormantParameters(200, 80, -3),
                F2=FormantParameters(1200, 120, -9),
                F3=FormantParameters(2600, 160, -15),
                F4=FormantParameters(3600, 200, -21),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.1
            ),
            'R': PhonemeFormants(  # rat
                F1=FormantParameters(200, 80, -3),
                F2=FormantParameters(1200, 120, -9),
                F3=FormantParameters(1690, 140, -12),
                F4=FormantParameters(3300, 180, -18),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            
            # Glides
            'W': PhonemeFormants(  # wet
                F1=FormantParameters(200, 80, -3),
                F2=FormantParameters(600, 100, -9),
                F3=FormantParameters(2200, 140, -15),
                F4=FormantParameters(3200, 180, -21),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            'Y': PhonemeFormants(  # yet
                F1=FormantParameters(200, 80, -3),
                F2=FormantParameters(2200, 140, -9),
                F3=FormantParameters(3000, 160, -15),
                F4=FormantParameters(3700, 200, -21),
                voicing=1.0, aspiration=0.0, frication=0.0, nasality=0.0
            ),
            
            # Silence
            'SIL': PhonemeFormants(
                F1=FormantParameters(0, 0, -60),
                F2=FormantParameters(0, 0, -60),
                F3=FormantParameters(0, 0, -60),
                F4=FormantParameters(0, 0, -60),
                voicing=0.0, aspiration=0.0, frication=0.0, nasality=0.0
            )
        }
    
    def synthesize_phoneme(self, phoneme: str, duration: float, f0: float, 
                          voice_params: Dict = None) -> np.ndarray:
        """
        Synthesize a single phoneme using Klatt formant synthesis
        
        Args:
            phoneme: ARPABET phoneme symbol
            duration: Duration in seconds
            f0: Fundamental frequency in Hz
            voice_params: Voice-specific parameters
            
        Returns:
            Synthesized audio samples
        """
        if voice_params is None:
            voice_params = {}
        
        # Get formant parameters for this phoneme
        if phoneme.upper() not in self.formant_data:
            phoneme = 'AH'  # Default to schwa
        
        formants = self.formant_data[phoneme.upper()]
        
        # Calculate number of samples
        samples = int(duration * self.sample_rate)
        if samples == 0:
            return np.array([])
        
        # Generate source signal
        source = self._generate_source_signal(samples, f0, formants, voice_params)
        
        # Apply formant filtering
        output = self._apply_formant_filters(source, formants, voice_params)
        
        # Apply envelope
        output = self._apply_phoneme_envelope(output, formants)
        
        return output
    
    def _generate_source_signal(self, samples: int, f0: float, 
                               formants: PhonemeFormants, voice_params: Dict) -> np.ndarray:
        """Generate the source signal (glottal pulses + noise)"""
        t = np.linspace(0, samples / self.sample_rate, samples)
        
        # Initialize source
        source = np.zeros(samples)
        
        # Voiced component (glottal pulses)
        if formants.voicing > 0 and f0 > 0:
            # Generate glottal pulse train
            glottal_pulses = self._generate_glottal_pulses(t, f0, formants.voicing)
            source += glottal_pulses * formants.voicing
        
        # Aspiration noise
        if formants.aspiration > 0:
            aspiration_noise = self._generate_aspiration_noise(samples)
            source += aspiration_noise * formants.aspiration
        
        # Frication noise
        if formants.frication > 0:
            frication_noise = self._generate_frication_noise(samples)
            source += frication_noise * formants.frication
        
        return source
    
    def _generate_glottal_pulses(self, t: np.ndarray, f0: float, voicing: float) -> np.ndarray:
        """Generate realistic glottal pulse train"""
        # Create glottal pulse shape (Rosenberg model)
        pulse_period = 1.0 / f0
        samples_per_period = int(pulse_period * self.sample_rate)
        
        if samples_per_period < 4:
            # F0 too high, use simple sawtooth
            return signal.sawtooth(2 * np.pi * f0 * t) * voicing
        
        # Generate one glottal pulse
        pulse_t = np.linspace(0, 1, samples_per_period)
        
        # Rosenberg glottal pulse model
        # Rising phase (40% of period)
        rising_samples = int(0.4 * samples_per_period)
        # Falling phase (60% of period)
        falling_samples = samples_per_period - rising_samples
        
        pulse = np.zeros(samples_per_period)
        
        # Rising phase: parabolic rise
        if rising_samples > 0:
            rise_t = np.linspace(0, 1, rising_samples)
            pulse[:rising_samples] = 0.5 * (1 - np.cos(np.pi * rise_t))
        
        # Falling phase: exponential decay
        if falling_samples > 0:
            fall_t = np.linspace(0, 1, falling_samples)
            pulse[rising_samples:] = np.exp(-3 * fall_t)
        
        # Repeat pulse for entire duration
        num_periods = len(t) // samples_per_period + 1
        full_pulse_train = np.tile(pulse, num_periods)[:len(t)]
        
        # Add slight jitter for naturalness
        jitter = np.random.normal(0, 0.02, len(t))
        full_pulse_train *= (1 + jitter)
        
        return full_pulse_train * voicing
    
    def _generate_aspiration_noise(self, samples: int) -> np.ndarray:
        """Generate aspiration noise (filtered white noise)"""
        # White noise
        noise = np.random.normal(0, 1, samples)
        
        # Filter to simulate aspiration spectrum
        # High-pass filter to emphasize higher frequencies
        if self.sample_rate > 1000:
            cutoff = 500 / self.nyquist  # 500 Hz highpass
            if cutoff < 1.0:
                b, a = signal.butter(2, cutoff, btype='highpass')
                noise = signal.filtfilt(b, a, noise)
        
        return noise * 0.3
    
    def _generate_frication_noise(self, samples: int) -> np.ndarray:
        """Generate frication noise (filtered noise for fricatives)"""
        # White noise
        noise = np.random.normal(0, 1, samples)
        
        # Filter to simulate frication spectrum
        # Bandpass filter in mid-high frequencies
        if self.sample_rate > 2000:
            low_freq = 2000 / self.nyquist
            high_freq = 8000 / self.nyquist
            if low_freq < high_freq and high_freq < 1.0:
                b, a = signal.butter(2, [low_freq, high_freq], btype='bandpass')
                noise = signal.filtfilt(b, a, noise)
        
        return noise * 0.4
    
    def _apply_formant_filters(self, source: np.ndarray, formants: PhonemeFormants, 
                              voice_params: Dict) -> np.ndarray:
        """Apply formant filtering to source signal"""
        if len(source) == 0:
            return source
        
        output = source.copy()
        
        # Apply each formant as a resonant filter
        formant_list = [formants.F1, formants.F2, formants.F3, formants.F4]
        
        for formant in formant_list:
            if formant.frequency > 50 and formant.frequency < self.nyquist - 50:
                # Create resonant filter
                output = self._apply_single_formant(output, formant, voice_params)
        
        return output
    
    def _apply_single_formant(self, signal_in: np.ndarray, formant: FormantParameters, 
                             voice_params: Dict) -> np.ndarray:
        """Apply a single formant filter"""
        # Convert amplitude from dB to linear
        amplitude = 10 ** (formant.amplitude / 20.0)
        
        # Apply voice-specific formant shifts
        freq = formant.frequency
        if 'formant_shifts' in voice_params:
            if formant.frequency < 800:  # F1 range
                freq *= voice_params['formant_shifts'].get('F1', 1.0)
            elif formant.frequency < 2000:  # F2 range
                freq *= voice_params['formant_shifts'].get('F2', 1.0)
            else:  # F3+ range
                freq *= voice_params['formant_shifts'].get('F3', 1.0)
        
        # Ensure frequency is valid
        if freq <= 50 or freq >= self.nyquist - 50:
            return signal_in
        
        # Create bandpass filter for formant
        bandwidth = formant.bandwidth
        low_freq = max(50, freq - bandwidth/2) / self.nyquist
        high_freq = min(self.nyquist - 50, freq + bandwidth/2) / self.nyquist
        
        if low_freq >= high_freq or high_freq >= 1.0:
            return signal_in
        
        try:
            # Design bandpass filter
            b, a = signal.butter(2, [low_freq, high_freq], btype='bandpass')
            
            # Apply filter
            formant_component = signal.filtfilt(b, a, signal_in)
            
            # Add formant component with appropriate amplitude
            return signal_in + formant_component * amplitude
            
        except Exception:
            # If filter design fails, return original signal
            return signal_in
    
    def _apply_phoneme_envelope(self, signal_in: np.ndarray, 
                               formants: PhonemeFormants) -> np.ndarray:
        """Apply phoneme-specific amplitude envelope"""
        if len(signal_in) == 0:
            return signal_in
        
        # Create basic envelope
        envelope = np.ones(len(signal_in))
        
        # Fade in/out for consonants
        if formants.voicing < 1.0:  # Consonant
            fade_samples = min(len(signal_in) // 4, int(0.02 * self.sample_rate))
            if fade_samples > 0:
                # Fade in
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                # Fade out
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        # Apply nasal coupling (reduces overall amplitude)
        if formants.nasality > 0:
            envelope *= (1.0 - 0.3 * formants.nasality)
        
        return signal_in * envelope
    
    def synthesize_diphone_transition(self, phoneme1: str, phoneme2: str, 
                                    duration: float, f0_contour: np.ndarray,
                                    voice_params: Dict = None) -> np.ndarray:
        """
        Synthesize a diphone with smooth formant transitions
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme  
            duration: Total duration in seconds
            f0_contour: Pitch contour over time
            voice_params: Voice-specific parameters
            
        Returns:
            Synthesized diphone audio
        """
        if voice_params is None:
            voice_params = {}
        
        samples = int(duration * self.sample_rate)
        if samples == 0:
            return np.array([])
        
        # Get formant data for both phonemes
        formants1 = self.formant_data.get(phoneme1.upper(), self.formant_data['AH'])
        formants2 = self.formant_data.get(phoneme2.upper(), self.formant_data['AH'])
        
        # Create time vector
        t = np.linspace(0, duration, samples)
        
        # Interpolate formant parameters over time
        transition_point = 0.5  # Transition at midpoint
        
        # Generate source signal with varying characteristics
        source = self._generate_transitioning_source(t, f0_contour, formants1, formants2, 
                                                   transition_point, voice_params)
        
        # Apply time-varying formant filtering
        output = self._apply_time_varying_formants(source, t, formants1, formants2, 
                                                 transition_point, voice_params)
        
        return output
    
    def _generate_transitioning_source(self, t: np.ndarray, f0_contour: np.ndarray,
                                     formants1: PhonemeFormants, formants2: PhonemeFormants,
                                     transition_point: float, voice_params: Dict) -> np.ndarray:
        """Generate source signal that transitions between two phonemes"""
        samples = len(t)
        source = np.zeros(samples)
        
        # Interpolate voicing characteristics
        transition_samples = int(transition_point * samples)
        
        for i, time_val in enumerate(t):
            # Interpolation factor (0 = phoneme1, 1 = phoneme2)
            if i < transition_samples:
                alpha = i / transition_samples if transition_samples > 0 else 0
            else:
                alpha = 1.0
            
            # Interpolate characteristics
            voicing = formants1.voicing * (1 - alpha) + formants2.voicing * alpha
            aspiration = formants1.aspiration * (1 - alpha) + formants2.aspiration * alpha
            frication = formants1.frication * (1 - alpha) + formants2.frication * alpha
            
            # Get F0 for this time point
            f0 = f0_contour[i] if i < len(f0_contour) else f0_contour[-1]
            
            # Generate source components
            sample_source = 0.0
            
            # Voiced component
            if voicing > 0 and f0 > 0:
                glottal_phase = 2 * np.pi * f0 * time_val
                glottal_pulse = self._single_glottal_pulse(glottal_phase)
                sample_source += glottal_pulse * voicing
            
            # Aspiration
            if aspiration > 0:
                sample_source += np.random.normal(0, 0.3) * aspiration
            
            # Frication
            if frication > 0:
                sample_source += np.random.normal(0, 0.4) * frication
            
            source[i] = sample_source
        
        return source
    
    def _single_glottal_pulse(self, phase: float) -> float:
        """Generate a single glottal pulse value at given phase"""
        # Normalize phase to [0, 2π]
        phase = phase % (2 * np.pi)
        
        # Rosenberg pulse model
        if phase < 0.4 * 2 * np.pi:  # Rising phase
            t_norm = phase / (0.4 * 2 * np.pi)
            return 0.5 * (1 - np.cos(np.pi * t_norm))
        else:  # Falling phase
            t_norm = (phase - 0.4 * 2 * np.pi) / (0.6 * 2 * np.pi)
            return np.exp(-3 * t_norm)
    
    def _apply_time_varying_formants(self, source: np.ndarray, t: np.ndarray,
                                   formants1: PhonemeFormants, formants2: PhonemeFormants,
                                   transition_point: float, voice_params: Dict) -> np.ndarray:
        """Apply formant filtering that varies over time"""
        if len(source) == 0:
            return source
        
        # For simplicity, apply formants in segments
        # In a full implementation, this would use time-varying filters
        
        transition_samples = int(transition_point * len(source))
        
        # First half - use formants1
        if transition_samples > 0:
            segment1 = source[:transition_samples]
            filtered1 = self._apply_formant_filters(segment1, formants1, voice_params)
        else:
            filtered1 = np.array([])
        
        # Second half - use formants2
        if transition_samples < len(source):
            segment2 = source[transition_samples:]
            filtered2 = self._apply_formant_filters(segment2, formants2, voice_params)
        else:
            filtered2 = np.array([])
        
        # Combine segments
        if len(filtered1) > 0 and len(filtered2) > 0:
            return np.concatenate([filtered1, filtered2])
        elif len(filtered1) > 0:
            return filtered1
        else:
            return filtered2
