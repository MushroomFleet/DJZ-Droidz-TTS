"""
LPC (Linear Predictive Coding) Synthesizer
Implements the core LPC synthesis used in DECtalk for authentic speech generation
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class LPCParameters:
    """LPC parameters for a phoneme"""
    coefficients: np.ndarray    # LPC coefficients
    gain: float                 # Excitation gain
    pitch_period: int           # Pitch period in samples (0 for unvoiced)
    voicing: float             # Voicing strength (0.0-1.0)
    duration: float            # Phoneme duration in seconds

class DECtalkLPCSynthesizer:
    """
    LPC synthesizer based on DECtalk's methodology
    Focuses on intelligibility with authentic 1996 characteristics
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.lpc_order = 16  # Increased from 12 to 16 for better spectral modeling
        self.lpc_database = self._create_lpc_database()
        
    def _create_lpc_database(self) -> Dict[str, LPCParameters]:
        """Create LPC parameter database with proper formant-derived coefficients"""
        return {
            # Vowels - optimized durations and enhanced formant precision for clarity
            'AA': LPCParameters(  # father [ɑ] - F1=730, F2=1090, F3=2440
                coefficients=self._formants_to_lpc([730, 1090, 2440], [70, 80, 110]),
                gain=1.1, pitch_period=180, voicing=1.0, duration=0.15
            ),
            'AE': LPCParameters(  # cat [æ] - F1=660, F2=1720, F3=2410
                coefficients=self._formants_to_lpc([660, 1720, 2410], [70, 80, 110]),
                gain=1.1, pitch_period=180, voicing=1.0, duration=0.14
            ),
            'AH': LPCParameters(  # cut [ʌ] - F1=520, F2=1190, F3=2390
                coefficients=self._formants_to_lpc([520, 1190, 2390], [60, 70, 100]),
                gain=1.0, pitch_period=180, voicing=1.0, duration=0.12
            ),
            'AO': LPCParameters(  # caught [ɔ] - F1=570, F2=840, F3=2410
                coefficients=self._formants_to_lpc([570, 840, 2410], [70, 70, 110]),
                gain=1.1, pitch_period=180, voicing=1.0, duration=0.15
            ),
            'EH': LPCParameters(  # red [ɛ] - F1=530, F2=1840, F3=2480
                coefficients=self._formants_to_lpc([530, 1840, 2480], [60, 80, 110]),
                gain=1.1, pitch_period=180, voicing=1.0, duration=0.13
            ),
            'ER': LPCParameters(  # hurt [ɝ] - F1=490, F2=1350, F3=1690
                coefficients=self._formants_to_lpc([490, 1350, 1690], [60, 80, 90]),
                gain=1.0, pitch_period=180, voicing=1.0, duration=0.16
            ),
            'IH': LPCParameters(  # hit [ɪ] - F1=390, F2=1990, F3=2550
                coefficients=self._formants_to_lpc([390, 1990, 2550], [50, 80, 110]),
                gain=1.0, pitch_period=180, voicing=1.0, duration=0.11
            ),
            'IY': LPCParameters(  # heat [i] - F1=270, F2=2290, F3=3010
                coefficients=self._formants_to_lpc([270, 2290, 3010], [40, 80, 120]),
                gain=1.0, pitch_period=180, voicing=1.0, duration=0.14
            ),
            'OW': LPCParameters(  # boat [o] - F1=360, F2=640, F3=2240
                coefficients=self._formants_to_lpc([360, 640, 2240], [50, 60, 100]),
                gain=1.1, pitch_period=180, voicing=1.0, duration=0.16
            ),
            'UH': LPCParameters(  # book [ʊ] - F1=440, F2=1020, F3=2240
                coefficients=self._formants_to_lpc([440, 1020, 2240], [60, 70, 100]),
                gain=1.0, pitch_period=180, voicing=1.0, duration=0.12
            ),
            'UW': LPCParameters(  # boot [u] - F1=300, F2=870, F3=2240
                coefficients=self._formants_to_lpc([300, 870, 2240], [50, 70, 100]),
                gain=1.0, pitch_period=180, voicing=1.0, duration=0.15
            ),
            'AW': LPCParameters(  # how [aʊ] - F1=570, F2=1030, F3=2380
                coefficients=self._formants_to_lpc([570, 1030, 2380], [70, 70, 110]),
                gain=1.1, pitch_period=180, voicing=1.0, duration=0.18
            ),
            'AY': LPCParameters(  # my [aɪ] - F1=570, F2=1400, F3=2500
                coefficients=self._formants_to_lpc([570, 1400, 2500], [70, 80, 110]),
                gain=1.1, pitch_period=180, voicing=1.0, duration=0.18
            ),
            'EY': LPCParameters(  # say [eɪ] - F1=390, F2=2000, F3=2550
                coefficients=self._formants_to_lpc([390, 2000, 2550], [50, 80, 110]),
                gain=1.1, pitch_period=180, voicing=1.0, duration=0.17
            ),
            'OY': LPCParameters(  # boy [ɔɪ] - F1=570, F2=840, F3=2410
                coefficients=self._formants_to_lpc([570, 840, 2410], [70, 70, 110]),
                gain=1.1, pitch_period=180, voicing=1.0, duration=0.17
            ),
            
            # Voiced Consonants - enhanced for clarity
            'B': LPCParameters(  # bat - voiced bilabial stop
                coefficients=self._formants_to_lpc([200, 1000, 2500], [100, 150, 200]),
                gain=0.8, pitch_period=180, voicing=0.9, duration=0.09
            ),
            'D': LPCParameters(  # dog - voiced alveolar stop
                coefficients=self._formants_to_lpc([300, 1700, 2600], [100, 150, 200]),
                gain=0.8, pitch_period=180, voicing=0.9, duration=0.09
            ),
            'G': LPCParameters(  # got - voiced velar stop
                coefficients=self._formants_to_lpc([250, 2300, 3000], [120, 180, 250]),
                gain=0.8, pitch_period=180, voicing=0.9, duration=0.09
            ),
            'V': LPCParameters(  # vat - voiced labiodental fricative
                coefficients=self._formants_to_lpc([200, 1000, 2500], [150, 200, 300]),
                gain=0.6, pitch_period=180, voicing=0.8, duration=0.12
            ),
            'Z': LPCParameters(  # zoo - voiced alveolar fricative
                coefficients=self._formants_to_lpc([200, 1700, 2600], [150, 200, 300]),
                gain=0.6, pitch_period=180, voicing=0.8, duration=0.12
            ),
            'ZH': LPCParameters(  # measure - voiced postalveolar fricative
                coefficients=self._formants_to_lpc([200, 1500, 2300], [150, 200, 300]),
                gain=0.6, pitch_period=180, voicing=0.8, duration=0.12
            ),
            'DH': LPCParameters(  # this - voiced dental fricative
                coefficients=self._formants_to_lpc([200, 1400, 2600], [150, 200, 300]),
                gain=0.5, pitch_period=180, voicing=0.7, duration=0.10
            ),
            'JH': LPCParameters(  # judge - voiced postalveolar affricate
                coefficients=self._formants_to_lpc([200, 1500, 2300], [120, 180, 250]),
                gain=0.7, pitch_period=180, voicing=0.8, duration=0.11
            ),
            'CH': LPCParameters(  # church - voiceless postalveolar affricate
                coefficients=self._formants_to_lpc([0, 1500, 2300], [0, 300, 400]),
                gain=0.5, pitch_period=0, voicing=0.0, duration=0.10
            ),
            
            # Unvoiced Consonants - enhanced clarity and timing
            'P': LPCParameters(  # pat - voiceless bilabial stop
                coefficients=self._formants_to_lpc([0, 1000, 2500], [0, 400, 500]),
                gain=0.5, pitch_period=0, voicing=0.0, duration=0.07
            ),
            'T': LPCParameters(  # top - voiceless alveolar stop
                coefficients=self._formants_to_lpc([0, 1700, 2600], [0, 400, 500]),
                gain=0.5, pitch_period=0, voicing=0.0, duration=0.07
            ),
            'K': LPCParameters(  # cat - voiceless velar stop
                coefficients=self._formants_to_lpc([0, 2300, 3000], [0, 500, 600]),
                gain=0.5, pitch_period=0, voicing=0.0, duration=0.07
            ),
            'F': LPCParameters(  # fat - voiceless labiodental fricative
                coefficients=self._formants_to_lpc([0, 1200, 2500], [0, 300, 400]),
                gain=0.4, pitch_period=0, voicing=0.0, duration=0.14
            ),
            'S': LPCParameters(  # sat - voiceless alveolar fricative
                coefficients=self._formants_to_lpc([0, 4000, 8000], [0, 500, 1000]),
                gain=0.4, pitch_period=0, voicing=0.0, duration=0.14
            ),
            'SH': LPCParameters(  # shot - voiceless postalveolar fricative
                coefficients=self._formants_to_lpc([0, 2000, 4000], [0, 400, 800]),
                gain=0.4, pitch_period=0, voicing=0.0, duration=0.14
            ),
            'TH': LPCParameters(  # think - voiceless dental fricative
                coefficients=self._formants_to_lpc([0, 1400, 2600], [0, 300, 400]),
                gain=0.3, pitch_period=0, voicing=0.0, duration=0.12
            ),
            'HH': LPCParameters(  # hat - voiceless glottal fricative
                coefficients=self._formants_to_lpc([0, 1500, 2500], [0, 500, 800]),
                gain=0.25, pitch_period=0, voicing=0.0, duration=0.09
            ),
            
            # Nasals - clear nasal resonance
            'M': LPCParameters(  # mat - bilabial nasal
                coefficients=np.array([1.0, -1.1, 0.4, -0.1, 0.03, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                gain=0.7, pitch_period=180, voicing=1.0, duration=0.10
            ),
            'N': LPCParameters(  # not - alveolar nasal
                coefficients=np.array([1.0, -1.0, 0.5, -0.15, 0.04, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                gain=0.7, pitch_period=180, voicing=1.0, duration=0.10
            ),
            'NG': LPCParameters(  # sing - velar nasal
                coefficients=np.array([1.0, -1.2, 0.3, -0.08, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                gain=0.7, pitch_period=180, voicing=1.0, duration=0.10
            ),
            
            # Liquids - smooth transitions
            'L': LPCParameters(  # lot - lateral liquid
                coefficients=np.array([1.0, -1.0, 0.6, -0.2, 0.06, -0.02, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0]),
                gain=0.7, pitch_period=180, voicing=1.0, duration=0.10
            ),
            'R': LPCParameters(  # rat - rhotic liquid
                coefficients=np.array([1.0, -1.2, 0.7, -0.3, 0.12, -0.05, 0.02, -0.008, 0.003, 0.0, 0.0, 0.0]),
                gain=0.7, pitch_period=180, voicing=1.0, duration=0.10
            ),
            
            # Glides - smooth formant transitions
            'W': LPCParameters(  # wet - labial-velar glide
                coefficients=np.array([1.0, -1.4, 0.4, -0.1, 0.03, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                gain=0.6, pitch_period=180, voicing=1.0, duration=0.08
            ),
            'Y': LPCParameters(  # yet - palatal glide
                coefficients=np.array([1.0, -0.8, 0.7, -0.25, 0.08, -0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0]),
                gain=0.6, pitch_period=180, voicing=1.0, duration=0.08
            ),
            
            # Silence
            'SIL': LPCParameters(
                coefficients=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                gain=0.0, pitch_period=0, voicing=0.0, duration=0.05
            )
        }
    
    def _formants_to_lpc(self, formant_freqs: List[float], bandwidths: List[float]) -> np.ndarray:
        """
        Convert formant frequencies and bandwidths to LPC coefficients
        This is the key to intelligible speech synthesis
        """
        # Convert formant frequencies to normalized frequencies
        nyquist = self.sample_rate / 2
        
        # Initialize with identity filter
        lpc_coeffs = np.array([1.0] + [0.0] * self.lpc_order)
        
        # Add each formant as a resonant pole pair
        for i, (freq, bw) in enumerate(zip(formant_freqs[:3], bandwidths[:3])):  # Use first 3 formants
            if freq > 0 and freq < nyquist:
                # Convert to normalized frequency
                omega = 2 * np.pi * freq / self.sample_rate
                
                # Calculate pole radius from bandwidth
                r = np.exp(-np.pi * bw / self.sample_rate)
                
                # Complex pole pair: r * exp(±jω)
                # This creates the resonance at the formant frequency
                pole_real = r * np.cos(omega)
                pole_imag = r * np.sin(omega)
                
                # Convert pole pair to polynomial coefficients
                # For poles at r*e^(±jω), the polynomial is:
                # (1 - r*e^(jω)*z^-1)(1 - r*e^(-jω)*z^-1) = 1 - 2*r*cos(ω)*z^-1 + r^2*z^-2
                a1 = -2 * pole_real
                a2 = r * r
                
                # Convolve with existing coefficients to add this formant
                formant_coeffs = np.array([1.0, a1, a2])
                
                # Extend arrays to same length for convolution
                max_len = len(lpc_coeffs) + len(formant_coeffs) - 1
                lpc_extended = np.pad(lpc_coeffs, (0, max_len - len(lpc_coeffs)))
                formant_extended = np.pad(formant_coeffs, (0, max_len - len(formant_coeffs)))
                
                # Convolve to combine filters
                new_coeffs = np.convolve(lpc_coeffs, formant_coeffs)
                
                # Truncate to desired order
                lpc_coeffs = new_coeffs[:self.lpc_order + 1]
        
        # Ensure stability by checking pole magnitudes
        lpc_coeffs = self._ensure_stability(lpc_coeffs)
        
        return lpc_coeffs
    
    def _ensure_stability(self, coeffs: np.ndarray) -> np.ndarray:
        """Ensure LPC filter stability by checking pole locations"""
        if len(coeffs) < 2:
            return coeffs
        
        # Find roots (poles) of the polynomial
        try:
            roots = np.roots(coeffs)
            
            # Check if any poles are outside unit circle
            unstable_poles = np.abs(roots) >= 1.0
            
            if np.any(unstable_poles):
                # Move unstable poles inside unit circle
                roots[unstable_poles] = roots[unstable_poles] / (np.abs(roots[unstable_poles]) * 1.01)
                
                # Reconstruct polynomial from modified roots
                stable_coeffs = np.poly(roots)
                
                # Normalize and truncate
                stable_coeffs = stable_coeffs / stable_coeffs[0]
                return stable_coeffs[:len(coeffs)]
        except:
            # If root finding fails, use original coefficients with reduced magnitude
            pass
        
        return coeffs
    
    def synthesize_phoneme(self, phoneme: str, f0: float = 122.0, 
                          voice_params: Dict = None) -> np.ndarray:
        """
        Synthesize a single phoneme using LPC synthesis
        
        Args:
            phoneme: ARPABET phoneme symbol
            f0: Fundamental frequency in Hz
            voice_params: Voice-specific parameters
            
        Returns:
            Synthesized audio samples
        """
        if voice_params is None:
            voice_params = {}
        
        # Get LPC parameters for this phoneme
        if phoneme.upper() not in self.lpc_database:
            phoneme = 'AH'  # Default to schwa
        
        lpc_params = self.lpc_database[phoneme.upper()]
        
        # Calculate number of samples
        duration = lpc_params.duration
        samples = int(duration * self.sample_rate)
        
        if samples == 0:
            return np.array([])
        
        # Generate excitation signal
        excitation = self._generate_excitation(samples, f0, lpc_params, voice_params)
        
        # Apply LPC synthesis filter
        output = self._apply_lpc_filter(excitation, lpc_params.coefficients, lpc_params.gain)
        
        # Apply DECtalk characteristics
        output = self._apply_dectalk_processing(output, phoneme, voice_params)
        
        return output
    
    def _generate_excitation(self, samples: int, f0: float, 
                           lpc_params: LPCParameters, voice_params: Dict) -> np.ndarray:
        """Generate enhanced excitation signal for LPC synthesis"""
        excitation = np.zeros(samples)
        
        if lpc_params.voicing > 0 and f0 > 0:
            # Voiced excitation - improved impulse train with glottal pulse shape
            pitch_period_samples = int(self.sample_rate / f0)
            
            if pitch_period_samples > 0:
                # Generate more realistic glottal pulse train
                for i in range(0, samples, pitch_period_samples):
                    if i < samples:
                        # Create glottal pulse with realistic shape
                        pulse_length = min(pitch_period_samples // 4, 20)  # Pulse duration
                        
                        for j in range(pulse_length):
                            if i + j < samples:
                                # Glottal pulse shape (exponential decay)
                                t = j / pulse_length
                                pulse_amplitude = lpc_params.voicing * np.exp(-3 * t) * (1 - t)
                                excitation[i + j] += pulse_amplitude
                
                # Add slight jitter for naturalness
                jitter_amount = voice_params.get('jitter', 0.02)
                if jitter_amount > 0:
                    jitter = np.random.normal(0, jitter_amount, samples)
                    excitation += jitter * lpc_params.voicing * 0.1
        
        # Enhanced noise component for unvoiced sounds
        noise_level = (1.0 - lpc_params.voicing) + voice_params.get('breathiness', 0.05)
        if noise_level > 0:
            # Generate filtered noise for more realistic unvoiced sounds
            noise = np.random.normal(0, 1, samples)
            
            # Apply high-pass filtering for fricatives
            if noise_level > 0.5:  # Likely unvoiced
                from scipy import signal
                try:
                    # High-pass filter for fricative-like noise
                    cutoff = 2000 / (self.sample_rate / 2)
                    b, a = signal.butter(2, cutoff, btype='highpass')
                    noise = signal.filtfilt(b, a, noise)
                except:
                    pass
            
            excitation += noise * noise_level * 0.15
        
        return excitation
    
    def _apply_lpc_filter(self, excitation: np.ndarray, coefficients: np.ndarray, 
                         gain: float) -> np.ndarray:
        """Apply LPC synthesis filter"""
        if len(excitation) == 0:
            return excitation
        
        # LPC filter is an all-pole filter: H(z) = G / (1 - sum(a_k * z^-k))
        # The denominator coefficients are the LPC coefficients
        denominator = coefficients[:self.lpc_order + 1]
        numerator = np.array([gain])
        
        # Apply the filter
        try:
            output = signal.lfilter(numerator, denominator, excitation)
        except:
            # If filter is unstable, use original excitation
            output = excitation * gain
        
        return output
    
    def _apply_dectalk_processing(self, waveform: np.ndarray, phoneme: str, 
                                voice_params: Dict) -> np.ndarray:
        """Apply DECtalk-specific processing for authenticity"""
        if len(waveform) == 0:
            return waveform
        
        result = waveform.copy()
        
        # 1. Apply characteristic DECtalk EQ
        result = self._apply_dectalk_eq(result)
        
        # 2. Add metallic resonance
        metallic_intensity = voice_params.get('timbre_metallic', 0.4)
        result = self._add_metallic_resonance(result, metallic_intensity)
        
        # 3. Apply envelope shaping
        result = self._apply_phoneme_envelope(result, phoneme)
        
        # 4. Normalize and limit
        result = self._normalize_and_limit(result)
        
        return result
    
    def _apply_dectalk_eq(self, waveform: np.ndarray) -> np.ndarray:
        """Apply characteristic DECtalk frequency response"""
        # Boost mid frequencies for intelligibility (1-3kHz)
        nyquist = self.sample_rate / 2
        low_freq = 1000 / nyquist
        high_freq = 3000 / nyquist
        
        if high_freq < 1.0 and low_freq < high_freq:
            try:
                b, a = signal.butter(2, [low_freq, high_freq], btype='bandpass')
                mid_boost = signal.filtfilt(b, a, waveform) * 0.15
                waveform = waveform + mid_boost
            except:
                pass
        
        # Slight high-frequency rolloff (characteristic of 1996 DACs)
        cutoff = 8000 / nyquist
        if cutoff < 1.0:
            try:
                b, a = signal.butter(2, cutoff, btype='lowpass')
                waveform = signal.filtfilt(b, a, waveform)
            except:
                pass
        
        return waveform
    
    def _add_metallic_resonance(self, waveform: np.ndarray, intensity: float) -> np.ndarray:
        """Add characteristic DECtalk metallic resonance"""
        if intensity <= 0:
            return waveform
        
        # Resonance around 3.2kHz (characteristic DECtalk frequency)
        resonant_freq = 3200
        nyquist = self.sample_rate / 2
        
        if resonant_freq < nyquist:
            center_freq = resonant_freq / nyquist
            bandwidth = 200 / nyquist
            
            low_freq = max(0.01, center_freq - bandwidth/2)
            high_freq = min(0.99, center_freq + bandwidth/2)
            
            if low_freq < high_freq:
                try:
                    b, a = signal.butter(2, [low_freq, high_freq], btype='bandpass')
                    metallic_component = signal.filtfilt(b, a, waveform) * intensity * 0.12
                    return waveform + metallic_component
                except:
                    pass
        
        return waveform
    
    def _apply_phoneme_envelope(self, waveform: np.ndarray, phoneme: str) -> np.ndarray:
        """Apply phoneme-specific amplitude envelope"""
        if len(waveform) == 0:
            return waveform
        
        envelope = np.ones(len(waveform))
        
        # Different envelope shapes for different phoneme types
        if phoneme in ['P', 'T', 'K', 'B', 'D', 'G']:  # Stops
            # Sharp attack, quick decay
            attack_samples = max(1, len(waveform) // 20)
            decay_samples = max(1, len(waveform) // 10)
            
            if attack_samples < len(waveform):
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            if decay_samples < len(waveform):
                envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
                
        elif phoneme in ['F', 'S', 'SH', 'TH', 'V', 'Z', 'ZH']:  # Fricatives
            # Gradual attack and decay
            fade_samples = max(1, len(waveform) // 8)
            
            if fade_samples < len(waveform):
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        return waveform * envelope
    
    def _normalize_and_limit(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize and apply soft limiting"""
        if len(waveform) == 0:
            return waveform
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.8
        
        # Apply soft limiting (characteristic of vintage hardware)
        threshold = 0.9
        over_threshold = np.abs(waveform) > threshold
        
        if np.any(over_threshold):
            # Soft limiting
            sign = np.sign(waveform)
            abs_wave = np.abs(waveform)
            limited = np.where(abs_wave > threshold,
                             threshold + (abs_wave - threshold) * 0.1,
                             abs_wave)
            waveform = sign * limited
        
        return waveform
    
    def synthesize_word(self, phonemes: List[str], f0: float = 122.0, 
                       voice_params: Dict = None) -> np.ndarray:
        """
        Synthesize a complete word from phoneme sequence
        
        Args:
            phonemes: List of ARPABET phoneme symbols
            f0: Base fundamental frequency
            voice_params: Voice-specific parameters
            
        Returns:
            Synthesized word audio
        """
        if not phonemes:
            return np.array([])
        
        if voice_params is None:
            voice_params = {}
        
        # Synthesize each phoneme
        phoneme_audio = []
        for i, phoneme in enumerate(phonemes):
            # Slight F0 variation for naturalness
            phoneme_f0 = f0 * (0.95 + 0.1 * np.random.random())
            
            audio = self.synthesize_phoneme(phoneme, phoneme_f0, voice_params)
            if len(audio) > 0:
                phoneme_audio.append(audio)
        
        if not phoneme_audio:
            return np.array([])
        
        # Concatenate with smooth transitions
        result = self._concatenate_with_transitions(phoneme_audio)
        
        # Add word-level processing
        result = self._apply_word_level_processing(result, voice_params)
        
        return result
    
    def _concatenate_with_transitions(self, phoneme_audio: List[np.ndarray]) -> np.ndarray:
        """Concatenate phonemes with smooth transitions"""
        if not phoneme_audio:
            return np.array([])
        
        if len(phoneme_audio) == 1:
            return phoneme_audio[0]
        
        result = phoneme_audio[0]
        
        for i in range(1, len(phoneme_audio)):
            current_audio = phoneme_audio[i]
            
            # Overlap and add for smooth transition
            overlap_samples = min(len(result), len(current_audio), 
                                int(0.01 * self.sample_rate))  # 10ms overlap
            
            if overlap_samples > 0:
                # Create crossfade
                fade_out = np.linspace(1, 0, overlap_samples)
                fade_in = np.linspace(0, 1, overlap_samples)
                
                # Apply crossfade
                result[-overlap_samples:] *= fade_out
                current_audio[:overlap_samples] *= fade_in
                
                # Add overlapped portion
                result[-overlap_samples:] += current_audio[:overlap_samples]
                
                # Append remaining portion
                if len(current_audio) > overlap_samples:
                    result = np.concatenate([result, current_audio[overlap_samples:]])
            else:
                result = np.concatenate([result, current_audio])
        
        return result
    
    def _apply_word_level_processing(self, waveform: np.ndarray, 
                                   voice_params: Dict) -> np.ndarray:
        """Apply word-level processing effects"""
        if len(waveform) == 0:
            return waveform
        
        result = waveform.copy()
        
        # Add slight amplitude modulation for naturalness
        t = np.linspace(0, len(result) / self.sample_rate, len(result))
        am_freq = 2.0  # 2 Hz modulation
        am_depth = 0.02  # 2% modulation depth
        am_signal = 1.0 + am_depth * np.sin(2 * np.pi * am_freq * t)
        result *= am_signal
        
        # Add characteristic DECtalk "click" at word boundaries
        if len(result) > int(0.02 * self.sample_rate):
            click_pos = int(0.01 * self.sample_rate)  # 10ms from start
            click_intensity = 0.1
            result[click_pos] += click_intensity * np.random.choice([-1, 1])
        
        return result
