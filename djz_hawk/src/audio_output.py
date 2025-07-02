"""
Audio Output Module for DJZ-Hawk
Handles audio playback and file saving with cross-platform compatibility
"""

import numpy as np
import wave
import sys
from typing import Optional
import tempfile
import os

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available, audio playback disabled")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available, using basic wave module")

class AudioOutput:
    """
    Cross-platform audio output handler for DJZ-Hawk
    Supports both playback and file saving
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.bit_depth = 16
        
    def play(self, waveform: np.ndarray, blocking: bool = True):
        """
        Play audio waveform through system audio
        
        Args:
            waveform: Audio data as numpy array
            blocking: Whether to wait for playback to complete
        """
        if not SOUNDDEVICE_AVAILABLE:
            print("Audio playback not available - sounddevice not installed")
            return
        
        if len(waveform) == 0:
            print("Warning: Empty waveform, nothing to play")
            return
        
        try:
            # Normalize audio to prevent clipping
            normalized_audio = self._normalize_audio(waveform)
            
            # Play audio
            sd.play(normalized_audio, samplerate=self.sample_rate)
            
            if blocking:
                sd.wait()  # Wait for playback to complete
                
        except Exception as e:
            print(f"Error during audio playback: {e}")
            # Fallback: try to save and play with system player
            self._fallback_play(waveform)
    
    def save_wav(self, waveform: np.ndarray, filename: str):
        """
        Save waveform to WAV file
        
        Args:
            waveform: Audio data as numpy array
            filename: Output filename
        """
        if len(waveform) == 0:
            print("Warning: Empty waveform, nothing to save")
            return
        
        try:
            # Normalize and convert to 16-bit
            normalized_audio = self._normalize_audio(waveform)
            audio_16bit = self._convert_to_16bit(normalized_audio)
            
            if SOUNDFILE_AVAILABLE:
                # Use soundfile for better format support
                sf.write(filename, audio_16bit, self.sample_rate, subtype='PCM_16')
            else:
                # Use basic wave module
                self._save_wav_basic(audio_16bit, filename)
                
            print(f"Audio saved to {filename}")
            
        except Exception as e:
            print(f"Error saving audio file: {e}")
            raise
    
    def _normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping"""
        if len(waveform) == 0:
            return waveform
        
        # Find peak amplitude
        peak = np.max(np.abs(waveform))
        
        if peak == 0:
            return waveform
        
        # Normalize to 90% of full scale to leave headroom
        normalized = waveform / peak * 0.9
        
        return normalized
    
    def _convert_to_16bit(self, waveform: np.ndarray) -> np.ndarray:
        """Convert floating point audio to 16-bit integer"""
        # Ensure values are in [-1, 1] range
        clipped = np.clip(waveform, -1.0, 1.0)
        
        # Convert to 16-bit integer
        audio_16bit = (clipped * 32767).astype(np.int16)
        
        return audio_16bit
    
    def _save_wav_basic(self, audio_16bit: np.ndarray, filename: str):
        """Save WAV file using basic wave module"""
        with wave.open(filename, 'wb') as wav_file:
            # Set WAV file parameters
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(self.sample_rate)
            
            # Write audio data
            wav_file.writeframes(audio_16bit.tobytes())
    
    def _fallback_play(self, waveform: np.ndarray):
        """Fallback audio playback using system player"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_filename = temp_file.name
            
            self.save_wav(waveform, temp_filename)
            
            # Try to play with system player
            if sys.platform.startswith('win'):
                # Windows
                os.system(f'start /wait "" "{temp_filename}"')
            elif sys.platform.startswith('darwin'):
                # macOS
                os.system(f'afplay "{temp_filename}"')
            elif sys.platform.startswith('linux'):
                # Linux - try common players
                players = ['aplay', 'paplay', 'play']
                for player in players:
                    if os.system(f'which {player} > /dev/null 2>&1') == 0:
                        os.system(f'{player} "{temp_filename}"')
                        break
                else:
                    print("No suitable audio player found on Linux")
            
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
                
        except Exception as e:
            print(f"Fallback audio playback failed: {e}")
    
    def get_audio_info(self, waveform: np.ndarray) -> dict:
        """Get information about audio waveform"""
        if len(waveform) == 0:
            return {
                'duration': 0.0,
                'samples': 0,
                'peak_amplitude': 0.0,
                'rms_amplitude': 0.0,
                'sample_rate': self.sample_rate
            }
        
        duration = len(waveform) / self.sample_rate
        peak_amplitude = np.max(np.abs(waveform))
        rms_amplitude = np.sqrt(np.mean(waveform ** 2))
        
        return {
            'duration': duration,
            'samples': len(waveform),
            'peak_amplitude': peak_amplitude,
            'rms_amplitude': rms_amplitude,
            'sample_rate': self.sample_rate
        }
    
    def apply_fade(self, waveform: np.ndarray, fade_in: float = 0.01, 
                   fade_out: float = 0.01) -> np.ndarray:
        """
        Apply fade in/out to audio waveform
        
        Args:
            waveform: Input audio
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            
        Returns:
            Audio with fades applied
        """
        if len(waveform) == 0:
            return waveform
        
        result = waveform.copy()
        
        # Fade in
        fade_in_samples = int(fade_in * self.sample_rate)
        if fade_in_samples > 0 and fade_in_samples < len(result):
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            result[:fade_in_samples] *= fade_in_curve
        
        # Fade out
        fade_out_samples = int(fade_out * self.sample_rate)
        if fade_out_samples > 0 and fade_out_samples < len(result):
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            result[-fade_out_samples:] *= fade_out_curve
        
        return result
    
    def mix_audio(self, waveforms: list, levels: Optional[list] = None) -> np.ndarray:
        """
        Mix multiple audio waveforms together
        
        Args:
            waveforms: List of audio waveforms
            levels: Optional list of mixing levels (0.0-1.0)
            
        Returns:
            Mixed audio
        """
        if not waveforms:
            return np.array([])
        
        if levels is None:
            levels = [1.0] * len(waveforms)
        
        # Find maximum length
        max_length = max(len(w) for w in waveforms)
        
        # Mix waveforms
        mixed = np.zeros(max_length)
        
        for waveform, level in zip(waveforms, levels):
            if len(waveform) > 0:
                # Pad waveform to max length
                padded = np.pad(waveform, (0, max_length - len(waveform)), 'constant')
                mixed += padded * level
        
        # Normalize to prevent clipping
        return self._normalize_audio(mixed)
