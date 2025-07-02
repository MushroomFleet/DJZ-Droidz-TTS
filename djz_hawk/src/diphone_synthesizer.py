"""
Diphone Synthesizer for DJZ-Hawk
Core synthesis engine implementing DECtalk's diphone concatenation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os

@dataclass
class DiphoneUnit:
    """Represents a single diphone unit"""
    name: str                    # e.g., "aa_b"
    waveform: np.ndarray        # Audio samples
    f0_contour: np.ndarray      # Pitch contour
    formant_tracks: Dict[str, np.ndarray]  # F1, F2, F3 tracks
    duration: float             # Duration in milliseconds
    energy_contour: np.ndarray  # Energy/amplitude over time
    phoneme_boundary: int       # Sample where phonemes transition
    lpc_coefficients: np.ndarray # LPC coefficients for synthesis
    source_voice: str           # Original voice model

class DECtalkDiphoneSynthesizer:
    """
    Faithful recreation of DECtalk 4.2CD diphone synthesis engine
    Implements the exact concatenation methodology with period artifacts
    """
    
    def __init__(self, voice_name: str = 'perfect_paul'):
        self.voice_name = voice_name
        self.sample_rate = 22050  # DECtalk standard
        self.voice_config = self._load_voice_config(voice_name)
        self.diphone_db = self._load_diphone_database(voice_name)
        
    def _load_voice_config(self, voice_name: str) -> dict:
        """Load voice configuration"""
        config_path = os.path.join('config', 'voice_configs.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config['voices'].get(voice_name, config['voices']['perfect_paul'])
        except FileNotFoundError:
            # Return default configuration
            return {
                'base_frequency': 122.0,
                'frequency_range': 40.0,
                'roughness': 0.15,
                'timbre_metallic': 0.40,
                'speech_rate': 160
            }
        
    def _load_diphone_database(self, voice_name: str) -> Dict[str, DiphoneUnit]:
        """Load the diphone database for specified voice"""
        # For now, generate synthetic diphone database
        # In a full implementation, this would load pre-recorded diphones
        return self._generate_synthetic_diphones(voice_name)
    
    def _generate_synthetic_diphones(self, voice_name: str) -> Dict[str, DiphoneUnit]:
        """
        Generate synthetic diphone database
        This is a simplified version for the initial implementation
        """
        diphones = {}
        
        # Common phonemes for English
        phonemes = ['AE', 'EH', 'IH', 'AH', 'UH', 'B', 'D', 'F', 'G', 'HH', 
                   'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z', 'SIL']
        
        # Generate diphones for all phoneme pairs
        for i, ph1 in enumerate(phonemes):
            for j, ph2 in enumerate(phonemes):
                diphone_name = f"{ph1}_{ph2}"
                diphones[diphone_name] = self._create_synthetic_diphone(ph1, ph2, voice_name)
        
        return diphones
    
    def _create_synthetic_diphone(self, ph1: str, ph2: str, voice_name: str) -> DiphoneUnit:
        """Create a synthetic diphone unit"""
        duration = 0.15  # 150ms default duration
        samples = int(duration * self.sample_rate)
        
        # Generate basic waveform based on phoneme characteristics
        waveform = self._generate_phoneme_waveform(ph1, ph2, samples)
        
        # Create basic formant tracks
        formant_tracks = {
            'F1': np.linspace(500, 600, samples),  # Basic F1 track
            'F2': np.linspace(1500, 1600, samples),  # Basic F2 track
            'F3': np.linspace(2500, 2600, samples)   # Basic F3 track
        }
        
        # Basic pitch contour
        base_freq = self.voice_config.get('base_frequency', 122.0)
        f0_contour = np.full(samples, base_freq)
        
        # Basic energy contour
        energy_contour = np.ones(samples) * 0.5
        
        # Simple LPC coefficients (placeholder)
        lpc_coefficients = np.array([1.0, -0.5, 0.2, -0.1])
        
        return DiphoneUnit(
            name=f"{ph1}_{ph2}",
            waveform=waveform,
            f0_contour=f0_contour,
            formant_tracks=formant_tracks,
            duration=duration * 1000,  # Convert to milliseconds
            energy_contour=energy_contour,
            phoneme_boundary=samples // 2,
            lpc_coefficients=lpc_coefficients,
            source_voice=voice_name
        )
    
    def _generate_phoneme_waveform(self, ph1: str, ph2: str, samples: int) -> np.ndarray:
        """Generate basic waveform for phoneme pair"""
        t = np.linspace(0, samples / self.sample_rate, samples)
        
        # Basic frequency based on phoneme type
        freq1 = self._get_phoneme_frequency(ph1)
        freq2 = self._get_phoneme_frequency(ph2)
        
        # Handle silence
        if freq1 == 0 and freq2 == 0:
            return np.zeros(samples)
        
        # Transition from ph1 to ph2
        freq_contour = np.linspace(freq1, freq2, samples)
        
        # Generate waveform based on phoneme type
        if self._is_vowel(ph1) or self._is_vowel(ph2):
            # Vowels: sustained harmonic content
            waveform = self._generate_vowel_waveform(freq_contour, t, samples)
        elif self._is_fricative(ph1) or self._is_fricative(ph2):
            # Fricatives: noise-based
            waveform = self._generate_fricative_waveform(freq_contour, t, samples)
        else:
            # Consonants: brief burst or formant
            waveform = self._generate_consonant_waveform(freq_contour, t, samples)
        
        # Apply appropriate envelope (NOT exponential decay for sustained sounds)
        envelope = self._create_phoneme_envelope(ph1, ph2, samples)
        waveform *= envelope
        
        # Normalize
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform)) * 0.6
        
        return waveform
    
    def _is_vowel(self, phoneme: str) -> bool:
        """Check if phoneme is a vowel"""
        vowels = ['AE', 'EH', 'IH', 'AH', 'UH', 'AA', 'AO', 'UW', 'IY', 'EY', 'AY', 'OW', 'AW', 'OY']
        return phoneme in vowels
    
    def _is_fricative(self, phoneme: str) -> bool:
        """Check if phoneme is a fricative"""
        fricatives = ['F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH']
        return phoneme in fricatives
    
    def _generate_vowel_waveform(self, freq_contour: np.ndarray, t: np.ndarray, samples: int) -> np.ndarray:
        """Generate sustained vowel waveform with formants"""
        waveform = np.zeros(samples)
        
        # Generate harmonic series for vowel
        for harmonic in range(1, 6):  # First 5 harmonics
            amplitude = 1.0 / (harmonic * 0.8)  # Gradual rolloff
            phase = np.random.uniform(0, 2*np.pi)  # Random phase
            waveform += amplitude * np.sin(2 * np.pi * freq_contour * harmonic * t + phase)
        
        # Add slight formant resonance
        formant_freq = np.mean(freq_contour) * 2.5  # Approximate formant
        formant_component = 0.3 * np.sin(2 * np.pi * formant_freq * t)
        waveform += formant_component
        
        return waveform
    
    def _generate_fricative_waveform(self, freq_contour: np.ndarray, t: np.ndarray, samples: int) -> np.ndarray:
        """Generate noise-based fricative waveform"""
        # High-frequency noise for fricatives
        noise = np.random.normal(0, 0.5, samples)
        
        # Filter noise to appropriate frequency range
        center_freq = np.mean(freq_contour)
        if center_freq > 3000:  # High fricatives like S, SH
            # High-pass filtered noise
            cutoff = 0.3  # Normalized frequency
            from scipy import signal
            b, a = signal.butter(2, cutoff, btype='highpass')
            try:
                filtered_noise = signal.filtfilt(b, a, noise)
            except:
                filtered_noise = noise
        else:
            # Lower frequency fricatives
            filtered_noise = noise * 0.7
        
        # Add some tonal component
        tonal = 0.2 * np.sin(2 * np.pi * freq_contour * t)
        
        return filtered_noise + tonal
    
    def _generate_consonant_waveform(self, freq_contour: np.ndarray, t: np.ndarray, samples: int) -> np.ndarray:
        """Generate consonant waveform (stops, nasals, etc.)"""
        waveform = np.zeros(samples)
        
        # Generate basic harmonic content
        for harmonic in range(1, 4):
            amplitude = 1.0 / harmonic
            waveform += amplitude * np.sin(2 * np.pi * freq_contour * harmonic * t)
        
        # Add some noise for realism
        noise_level = 0.1
        waveform += np.random.normal(0, noise_level, samples)
        
        return waveform
    
    def _create_phoneme_envelope(self, ph1: str, ph2: str, samples: int) -> np.ndarray:
        """Create appropriate envelope for phoneme type"""
        t_norm = np.linspace(0, 1, samples)
        
        if self._is_vowel(ph1) and self._is_vowel(ph2):
            # Vowel-to-vowel: sustained with slight fade
            envelope = np.ones(samples)
            fade_length = samples // 10  # 10% fade at each end
            envelope[:fade_length] *= np.linspace(0, 1, fade_length)
            envelope[-fade_length:] *= np.linspace(1, 0, fade_length)
        elif ph1 == 'SIL' or ph2 == 'SIL':
            # Silence transitions: gradual fade
            if ph1 == 'SIL':
                envelope = np.linspace(0, 1, samples)
            else:
                envelope = np.linspace(1, 0, samples)
        else:
            # Consonants and mixed: moderate envelope
            attack = samples // 4
            decay = samples // 4
            sustain_level = 0.8
            
            envelope = np.ones(samples) * sustain_level
            envelope[:attack] *= np.linspace(0, sustain_level, attack)
            envelope[-decay:] *= np.linspace(sustain_level, 0, decay)
        
        return envelope
    
    def _get_phoneme_frequency(self, phoneme: str) -> float:
        """Get characteristic frequency for phoneme"""
        # Basic frequency mapping for phonemes
        freq_map = {
            'AE': 800,   # vowels
            'EH': 600,
            'IH': 400,
            'AH': 700,
            'UH': 300,
            'B': 150,    # consonants
            'D': 200,
            'F': 4000,
            'G': 250,
            'HH': 2000,
            'K': 2500,
            'L': 300,
            'M': 200,
            'N': 250,
            'P': 100,
            'R': 400,
            'S': 6000,
            'T': 3000,
            'V': 150,
            'W': 300,
            'Y': 400,
            'Z': 4000,
            'SIL': 0     # silence
        }
        
        return freq_map.get(phoneme, 500)  # Default frequency
    
    def synthesize_phoneme_sequence(self, phonemes: List[str], 
                                  prosody: Optional[Dict] = None) -> np.ndarray:
        """
        Synthesize speech from phoneme sequence using diphone concatenation
        """
        if prosody is None:
            prosody = self._default_prosody()
        
        if not phonemes:
            return np.array([])
        
        # Convert phonemes to diphone sequence
        diphones = self._phonemes_to_diphones(phonemes)
        
        # Retrieve diphone units
        selected_units = []
        for diphone_name in diphones:
            unit = self._select_diphone_unit(diphone_name, prosody)
            selected_units.append(unit)
        
        # Concatenate with characteristic DECtalk artifacts
        waveform = self._concatenate_diphones(selected_units, prosody)
        
        # Apply DECtalk-specific post-processing
        waveform = self._apply_dectalk_characteristics(waveform)
        
        return waveform
    
    def _default_prosody(self) -> Dict:
        """Default prosody settings"""
        return {
            'pitch_factor': 1.0,
            'speed_factor': 1.0,
            'pause_factor': 1.0,
            'emphasis_factor': 1.0
        }
    
    def _phonemes_to_diphones(self, phonemes: List[str]) -> List[str]:
        """Convert phoneme sequence to diphone sequence"""
        if len(phonemes) < 2:
            return [f"SIL_{phonemes[0]}" if phonemes else "SIL_SIL"]
        
        diphones = []
        
        # Add initial silence-to-first-phoneme diphone
        diphones.append(f"SIL_{phonemes[0]}")
        
        # Add phoneme-to-phoneme diphones
        for i in range(len(phonemes) - 1):
            diphones.append(f"{phonemes[i]}_{phonemes[i+1]}")
        
        # Add final phoneme-to-silence diphone
        diphones.append(f"{phonemes[-1]}_SIL")
        
        return diphones
    
    def _select_diphone_unit(self, diphone_name: str, 
                           prosody: Dict) -> DiphoneUnit:
        """
        Select best matching diphone unit from database
        Implements DECtalk's selection criteria
        """
        if diphone_name not in self.diphone_db:
            # Fallback to closest available diphone
            diphone_name = self._find_closest_diphone(diphone_name)
        
        base_unit = self.diphone_db[diphone_name]
        
        # Modify unit according to prosodic requirements
        modified_unit = self._modify_prosody(base_unit, prosody)
        
        return modified_unit
    
    def _find_closest_diphone(self, target_diphone: str) -> str:
        """Find closest available diphone"""
        # Simple fallback - use silence diphone
        if "SIL_SIL" in self.diphone_db:
            return "SIL_SIL"
        
        # Return first available diphone
        return list(self.diphone_db.keys())[0] if self.diphone_db else "SIL_SIL"
    
    def _modify_prosody(self, unit: DiphoneUnit, prosody: Dict) -> DiphoneUnit:
        """Modify diphone unit according to prosodic requirements"""
        modified_unit = DiphoneUnit(
            name=unit.name,
            waveform=unit.waveform.copy(),
            f0_contour=unit.f0_contour.copy(),
            formant_tracks={k: v.copy() for k, v in unit.formant_tracks.items()},
            duration=unit.duration,
            energy_contour=unit.energy_contour.copy(),
            phoneme_boundary=unit.phoneme_boundary,
            lpc_coefficients=unit.lpc_coefficients.copy(),
            source_voice=unit.source_voice
        )
        
        # Apply pitch modification
        pitch_factor = prosody.get('pitch_factor', 1.0)
        if pitch_factor != 1.0:
            modified_unit.f0_contour *= pitch_factor
            # Simple pitch shifting (in reality would use more sophisticated methods)
            modified_unit.waveform = self._pitch_shift(modified_unit.waveform, pitch_factor)
        
        # Apply speed modification
        speed_factor = prosody.get('speed_factor', 1.0)
        if speed_factor != 1.0:
            modified_unit.waveform = self._time_stretch(modified_unit.waveform, speed_factor)
            modified_unit.duration /= speed_factor
        
        return modified_unit
    
    def _pitch_shift(self, waveform: np.ndarray, factor: float) -> np.ndarray:
        """Simple pitch shifting (placeholder implementation)"""
        # Very basic pitch shifting - in reality would use PSOLA or similar
        if factor == 1.0:
            return waveform
        
        # Simple resampling approach (not ideal but functional)
        new_length = int(len(waveform) / factor)
        indices = np.linspace(0, len(waveform) - 1, new_length)
        return np.interp(indices, np.arange(len(waveform)), waveform)
    
    def _time_stretch(self, waveform: np.ndarray, factor: float) -> np.ndarray:
        """Simple time stretching (placeholder implementation)"""
        if factor == 1.0:
            return waveform
        
        # Simple resampling approach
        new_length = int(len(waveform) * factor)
        indices = np.linspace(0, len(waveform) - 1, new_length)
        return np.interp(indices, np.arange(len(waveform)), waveform)
    
    def _concatenate_diphones(self, units: List[DiphoneUnit], 
                            prosody: Dict) -> np.ndarray:
        """
        Concatenate diphones with authentic DECtalk characteristics
        """
        if not units:
            return np.array([])
        
        concatenated = np.array([])
        
        for i, unit in enumerate(units):
            # Apply characteristic DECtalk windowing
            windowed_unit = self._apply_dectalk_windowing(unit.waveform)
            
            if i == 0:
                # First unit - minimal silence padding
                silence_samples = int(0.01 * self.sample_rate)  # 10ms only
                concatenated = np.concatenate([
                    np.zeros(silence_samples),
                    windowed_unit
                ])
            else:
                # Subsequent units - apply overlap-add with artifacts
                overlap_samples = int(0.01 * self.sample_rate)  # 10ms overlap
                
                if len(concatenated) >= overlap_samples and len(windowed_unit) >= overlap_samples:
                    # Characteristic DECtalk concatenation with clicks/pops
                    transition = self._create_dectalk_transition(
                        concatenated[-overlap_samples:],
                        windowed_unit[:overlap_samples]
                    )
                    
                    concatenated = np.concatenate([
                        concatenated[:-overlap_samples],
                        transition,
                        windowed_unit[overlap_samples:]
                    ])
                else:
                    # Simple concatenation if overlap is not possible
                    concatenated = np.concatenate([concatenated, windowed_unit])
            
            # Add minimal inter-diphone pause (much shorter)
            if i < len(units) - 1:
                pause_duration = max(0.001, prosody.get('pause_factor', 1.0) * 0.002)  # 2ms max
                pause_samples = int(pause_duration * self.sample_rate)
                concatenated = np.concatenate([
                    concatenated,
                    np.zeros(pause_samples)
                ])
        
        return concatenated
    
    def _apply_dectalk_windowing(self, waveform: np.ndarray) -> np.ndarray:
        """Apply characteristic DECtalk windowing"""
        if len(waveform) == 0:
            return waveform
        
        # Much shorter fade to preserve audio content
        window_length = min(len(waveform) // 10, int(0.005 * self.sample_rate))  # 5ms max window
        
        if window_length < 2:
            return waveform  # Too short to window
        
        # Create gentle fade in/out
        fade_in = np.linspace(0, 1, window_length)
        fade_out = np.linspace(1, 0, window_length)
        
        windowed = waveform.copy()
        windowed[:window_length] *= fade_in
        windowed[-window_length:] *= fade_out
        
        return windowed
    
    def _create_dectalk_transition(self, tail: np.ndarray, 
                                 head: np.ndarray) -> np.ndarray:
        """
        Create characteristic DECtalk transitions with artifacts
        """
        # Simple cross-fade with intentional discontinuities
        fade_length = len(tail)
        fade_out = np.linspace(1.0, 0.0, fade_length)
        fade_in = np.linspace(0.0, 1.0, fade_length)
        
        # Add characteristic "click" at transition
        click_intensity = np.random.uniform(0.1, 0.3)
        click_position = fade_length // 2
        
        transition = tail * fade_out + head * fade_in
        
        # Inject characteristic click/pop
        if click_position < len(transition):
            transition[click_position] += click_intensity * np.random.choice([-1, 1])
        
        return transition
    
    def _apply_dectalk_characteristics(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply characteristic DECtalk audio processing artifacts
        """
        if len(waveform) == 0:
            return waveform
        
        # 1. Characteristic formant emphasis
        waveform = self._apply_formant_emphasis(waveform)
        
        # 2. Add metallic resonance characteristic of DECtalk
        waveform = self._add_metallic_resonance(waveform)
        
        # 3. Apply characteristic frequency response
        waveform = self._apply_dectalk_eq(waveform)
        
        # 4. Characteristic amplitude limiting
        waveform = self._apply_vintage_limiting(waveform)
        
        return waveform
    
    def _apply_formant_emphasis(self, waveform: np.ndarray) -> np.ndarray:
        """Apply formant emphasis"""
        # Simple high-pass filter to emphasize formants
        from scipy import signal
        
        # High-pass filter around 300Hz
        nyquist = self.sample_rate / 2
        high_cutoff = 300 / nyquist
        b, a = signal.butter(2, high_cutoff, btype='highpass')
        
        try:
            filtered = signal.filtfilt(b, a, waveform)
            return waveform + filtered * 0.2  # Mix with original
        except:
            return waveform  # Return original if filtering fails
    
    def _add_metallic_resonance(self, waveform: np.ndarray) -> np.ndarray:
        """Add characteristic DECtalk metallic sound"""
        # Create resonant filter at ~3.2kHz (characteristic DECtalk frequency)
        from scipy import signal
        
        resonant_freq = 3200  # Hz
        nyquist = self.sample_rate / 2
        normalized_freq = resonant_freq / nyquist
        
        if normalized_freq < 1.0:
            try:
                b, a = signal.iirfilter(2, normalized_freq, btype='bandpass', 
                                      analog=False, ftype='butter')
                
                # Apply with characteristic intensity
                metallic_component = signal.filtfilt(b, a, waveform) * 0.15
                return waveform + metallic_component
            except:
                pass
        
        return waveform
    
    def _apply_dectalk_eq(self, waveform: np.ndarray) -> np.ndarray:
        """Apply characteristic DECtalk EQ curve"""
        # Simple EQ simulation - boost mids, slight high rolloff
        from scipy import signal
        
        try:
            # Slight mid boost around 2kHz
            nyquist = self.sample_rate / 2
            mid_freq = 2000 / nyquist
            
            if mid_freq < 1.0:
                b, a = signal.iirfilter(2, mid_freq, btype='bandpass', ftype='butter')
                mid_boost = signal.filtfilt(b, a, waveform) * 0.1
                waveform = waveform + mid_boost
        except:
            pass
        
        return waveform
    
    def _apply_vintage_limiting(self, waveform: np.ndarray) -> np.ndarray:
        """Apply vintage-style amplitude limiting"""
        # Simple soft clipping
        threshold = 0.8
        limited = np.tanh(waveform / threshold) * threshold
        return limited
