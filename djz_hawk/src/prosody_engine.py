"""
Prosody Engine for DJZ-Hawk
Handles stress, intonation, and rhythm patterns for DECtalk 4.2CD recreation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re

@dataclass
class ProsodyMarker:
    """Represents a prosodic marker in the text"""
    position: int           # Position in phoneme sequence
    type: str              # 'stress', 'pause', 'pitch_rise', 'pitch_fall'
    strength: float        # Intensity of the prosodic feature (0.0-1.0)
    duration: float        # Duration in seconds (for pauses)

@dataclass
class StressPattern:
    """Represents stress pattern for a word or phrase"""
    syllable_positions: List[int]  # Positions of syllable nuclei
    stress_levels: List[int]       # 0=unstressed, 1=primary, 2=secondary
    word_boundaries: List[int]     # Word boundary positions

class DECtalkProsodyEngine:
    """
    Prosody engine that recreates DECtalk 4.2CD's characteristic
    limited but distinctive prosodic patterns
    """
    
    def __init__(self):
        self.stress_rules = self._load_stress_rules()
        self.intonation_patterns = self._load_intonation_patterns()
        self.pause_rules = self._load_pause_rules()
        
    def _load_stress_rules(self) -> Dict:
        """Load English stress assignment rules"""
        return {
            # Common stress patterns for different word types
            'monosyllabic': {'pattern': [1], 'confidence': 0.9},
            'disyllabic_noun': {'pattern': [1, 0], 'confidence': 0.8},
            'disyllabic_verb': {'pattern': [0, 1], 'confidence': 0.7},
            'trisyllabic_initial': {'pattern': [1, 0, 0], 'confidence': 0.8},
            'trisyllabic_penult': {'pattern': [0, 1, 0], 'confidence': 0.7},
            'compound_word': {'pattern': [1, 0, 2, 0], 'confidence': 0.9},
            
            # Suffix-based rules
            'tion_suffix': {'pattern': [0, 1], 'confidence': 0.95},
            'ic_suffix': {'pattern': [1, 0], 'confidence': 0.9},
            'ity_suffix': {'pattern': [1, 0, 0], 'confidence': 0.9},
            'ment_suffix': {'pattern': [1, 0], 'confidence': 0.8},
        }
    
    def _load_intonation_patterns(self) -> Dict:
        """Load intonation patterns for different sentence types"""
        return {
            'declarative': {
                'initial_f0': 1.0,      # Relative to base frequency
                'peak_position': 0.3,   # Position of peak (0.0-1.0)
                'peak_height': 1.2,     # Peak height multiplier
                'final_fall': 0.8,      # Final frequency multiplier
                'slope': -0.1           # Overall declination slope
            },
            'interrogative': {
                'initial_f0': 1.0,
                'peak_position': 0.7,   # Later peak for questions
                'peak_height': 1.4,     # Higher peak
                'final_fall': 1.1,      # Rising final
                'slope': 0.05           # Slight rise
            },
            'exclamatory': {
                'initial_f0': 1.1,
                'peak_position': 0.2,   # Early peak
                'peak_height': 1.5,     # High peak
                'final_fall': 0.7,      # Strong final fall
                'slope': -0.15          # Steep declination
            },
            'list_item': {
                'initial_f0': 1.0,
                'peak_position': 0.4,
                'peak_height': 1.1,
                'final_fall': 0.95,     # Slight continuation rise
                'slope': -0.05
            }
        }
    
    def _load_pause_rules(self) -> Dict:
        """Load pause insertion rules"""
        return {
            'sentence_final': {'duration': 0.5, 'strength': 1.0},
            'clause_boundary': {'duration': 0.3, 'strength': 0.8},
            'phrase_boundary': {'duration': 0.2, 'strength': 0.6},
            'word_boundary': {'duration': 0.05, 'strength': 0.3},
            'comma': {'duration': 0.25, 'strength': 0.7},
            'semicolon': {'duration': 0.35, 'strength': 0.8},
            'colon': {'duration': 0.3, 'strength': 0.75},
            'question_mark': {'duration': 0.4, 'strength': 0.9},
            'exclamation': {'duration': 0.45, 'strength': 0.95}
        }
    
    def analyze_prosody(self, text: str, phonemes: List[str]) -> List[ProsodyMarker]:
        """
        Analyze text and phoneme sequence to generate prosodic markers
        """
        markers = []
        
        # 1. Detect sentence boundaries and types
        sentences = self._segment_sentences(text)
        sentence_markers = self._analyze_sentence_prosody(sentences, phonemes)
        markers.extend(sentence_markers)
        
        # 2. Detect stress patterns
        stress_markers = self._analyze_stress_patterns(text, phonemes)
        markers.extend(stress_markers)
        
        # 3. Insert pauses based on punctuation
        pause_markers = self._analyze_pause_patterns(text, phonemes)
        markers.extend(pause_markers)
        
        # 4. Apply DECtalk-specific prosodic limitations
        markers = self._apply_dectalk_limitations(markers)
        
        return sorted(markers, key=lambda x: x.position)
    
    def _segment_sentences(self, text: str) -> List[Tuple[str, str]]:
        """Segment text into sentences with their types"""
        sentences = []
        
        # Split on sentence-ending punctuation
        sentence_pattern = r'([.!?]+)'
        parts = re.split(sentence_pattern, text)
        
        current_sentence = ""
        for i, part in enumerate(parts):
            if re.match(r'[.!?]+', part):
                if current_sentence.strip():
                    sentence_type = self._classify_sentence_type(current_sentence, part)
                    sentences.append((current_sentence.strip(), sentence_type))
                current_sentence = ""
            else:
                current_sentence += part
        
        # Handle final sentence without punctuation
        if current_sentence.strip():
            sentences.append((current_sentence.strip(), 'declarative'))
        
        return sentences
    
    def _classify_sentence_type(self, sentence: str, punctuation: str) -> str:
        """Classify sentence type based on content and punctuation"""
        sentence_lower = sentence.lower().strip()
        
        if '?' in punctuation:
            return 'interrogative'
        elif '!' in punctuation:
            return 'exclamatory'
        elif sentence_lower.startswith(('what', 'where', 'when', 'why', 'how', 'who')):
            return 'interrogative'
        elif sentence_lower.startswith(('is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would')):
            return 'interrogative'
        else:
            return 'declarative'
    
    def _analyze_sentence_prosody(self, sentences: List[Tuple[str, str]], 
                                phonemes: List[str]) -> List[ProsodyMarker]:
        """Analyze sentence-level prosodic patterns"""
        markers = []
        phoneme_pos = 0
        
        for sentence_text, sentence_type in sentences:
            # Estimate phonemes for this sentence
            sentence_phoneme_count = len(sentence_text.split()) * 3  # Rough estimate
            sentence_end = min(phoneme_pos + sentence_phoneme_count, len(phonemes))
            
            # Apply intonation pattern
            pattern = self.intonation_patterns.get(sentence_type, 
                                                 self.intonation_patterns['declarative'])
            
            # Add pitch markers throughout sentence
            sentence_length = sentence_end - phoneme_pos
            if sentence_length > 0:
                # Initial pitch
                markers.append(ProsodyMarker(
                    position=phoneme_pos,
                    type='pitch_set',
                    strength=pattern['initial_f0'],
                    duration=0.0
                ))
                
                # Peak position
                peak_pos = phoneme_pos + int(sentence_length * pattern['peak_position'])
                markers.append(ProsodyMarker(
                    position=peak_pos,
                    type='pitch_peak',
                    strength=pattern['peak_height'],
                    duration=0.0
                ))
                
                # Final pitch
                markers.append(ProsodyMarker(
                    position=sentence_end - 1,
                    type='pitch_final',
                    strength=pattern['final_fall'],
                    duration=0.0
                ))
            
            phoneme_pos = sentence_end
        
        return markers
    
    def _analyze_stress_patterns(self, text: str, phonemes: List[str]) -> List[ProsodyMarker]:
        """Analyze word-level stress patterns"""
        markers = []
        words = text.split()
        phoneme_pos = 0
        
        for word in words:
            # Clean word of punctuation
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if not clean_word:
                continue
            
            # Estimate phonemes for this word
            word_phoneme_count = max(1, len(clean_word) // 2)  # Rough estimate
            word_end = min(phoneme_pos + word_phoneme_count, len(phonemes))
            
            # Determine stress pattern
            stress_pattern = self._get_word_stress_pattern(clean_word)
            
            # Apply stress to syllables
            if stress_pattern and word_end > phoneme_pos:
                syllable_positions = self._estimate_syllable_positions(
                    phoneme_pos, word_end, len(stress_pattern)
                )
                
                for i, (syll_pos, stress_level) in enumerate(zip(syllable_positions, stress_pattern)):
                    if stress_level > 0:
                        markers.append(ProsodyMarker(
                            position=syll_pos,
                            type='stress',
                            strength=stress_level / 2.0,  # Normalize to 0.0-1.0
                            duration=0.0
                        ))
            
            phoneme_pos = word_end
        
        return markers
    
    def _get_word_stress_pattern(self, word: str) -> List[int]:
        """Determine stress pattern for a word"""
        word_len = len(word)
        
        # Check for specific patterns
        if word.endswith('tion'):
            return [0, 1] if word_len > 4 else [1]
        elif word.endswith('ic'):
            return [1, 0] if word_len > 2 else [1]
        elif word.endswith('ity'):
            return [1, 0, 0] if word_len > 3 else [1, 0]
        elif word.endswith('ment'):
            return [1, 0] if word_len > 4 else [1]
        
        # Default patterns based on syllable count
        syllable_count = self._estimate_syllable_count(word)
        
        if syllable_count == 1:
            return [1]
        elif syllable_count == 2:
            # Most English disyllabic words stress first syllable
            return [1, 0]
        elif syllable_count == 3:
            # Most trisyllabic words stress first syllable
            return [1, 0, 0]
        else:
            # Longer words - stress first and third syllables
            pattern = [0] * syllable_count
            pattern[0] = 1  # Primary stress
            if syllable_count > 2:
                pattern[2] = 2  # Secondary stress
            return pattern
    
    def _estimate_syllable_count(self, word: str) -> int:
        """Rough estimate of syllable count"""
        vowels = 'aeiouy'
        word = word.lower()
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
    
    def _estimate_syllable_positions(self, start: int, end: int, syllable_count: int) -> List[int]:
        """Estimate positions of syllable nuclei"""
        if syllable_count <= 1:
            return [start + (end - start) // 2]
        
        positions = []
        segment_length = (end - start) / syllable_count
        
        for i in range(syllable_count):
            pos = start + int((i + 0.5) * segment_length)
            positions.append(min(pos, end - 1))
        
        return positions
    
    def _analyze_pause_patterns(self, text: str, phonemes: List[str]) -> List[ProsodyMarker]:
        """Analyze punctuation-based pause patterns"""
        markers = []
        
        # Find punctuation positions in text
        punctuation_pattern = r'([,.;:!?])'
        parts = re.split(punctuation_pattern, text)
        
        char_pos = 0
        phoneme_pos = 0
        
        for part in parts:
            if re.match(punctuation_pattern, part):
                # This is punctuation - add pause
                pause_type = self._classify_pause_type(part)
                pause_rule = self.pause_rules.get(pause_type, self.pause_rules['word_boundary'])
                
                markers.append(ProsodyMarker(
                    position=phoneme_pos,
                    type='pause',
                    strength=pause_rule['strength'],
                    duration=pause_rule['duration']
                ))
            else:
                # Regular text - advance phoneme position
                word_count = len(part.split())
                phoneme_advance = word_count * 3  # Rough estimate
                phoneme_pos = min(phoneme_pos + phoneme_advance, len(phonemes))
            
            char_pos += len(part)
        
        return markers
    
    def _classify_pause_type(self, punctuation: str) -> str:
        """Classify punctuation for pause rules"""
        if '.' in punctuation:
            return 'sentence_final'
        elif '?' in punctuation:
            return 'question_mark'
        elif '!' in punctuation:
            return 'exclamation'
        elif ';' in punctuation:
            return 'semicolon'
        elif ':' in punctuation:
            return 'colon'
        elif ',' in punctuation:
            return 'comma'
        else:
            return 'phrase_boundary'
    
    def _apply_dectalk_limitations(self, markers: List[ProsodyMarker]) -> List[ProsodyMarker]:
        """
        Apply DECtalk's characteristic prosodic limitations
        DECtalk had limited prosodic variation compared to natural speech
        """
        limited_markers = []
        
        for marker in markers:
            # Reduce prosodic variation to match DECtalk's limited range
            if marker.type == 'stress':
                # DECtalk had subtle stress differences
                marker.strength *= 0.6  # Reduce stress prominence
            elif marker.type in ['pitch_peak', 'pitch_final']:
                # Limit pitch range
                if marker.strength > 1.0:
                    marker.strength = 1.0 + (marker.strength - 1.0) * 0.5
                elif marker.strength < 1.0:
                    marker.strength = 1.0 - (1.0 - marker.strength) * 0.5
            elif marker.type == 'pause':
                # DECtalk had characteristic pause durations
                marker.duration = min(marker.duration, 0.6)  # Max pause duration
            
            limited_markers.append(marker)
        
        return limited_markers
    
    def apply_prosody_to_synthesis(self, waveform: np.ndarray, phonemes: List[str], 
                                 markers: List[ProsodyMarker], 
                                 base_f0: float = 122.0) -> np.ndarray:
        """
        Apply prosodic modifications to synthesized waveform
        """
        if len(waveform) == 0:
            return waveform
        
        sample_rate = 22050
        modified_waveform = waveform.copy()
        
        # Create time-to-phoneme mapping
        phoneme_duration = len(waveform) / len(phonemes) if phonemes else len(waveform)
        
        for marker in markers:
            if marker.position >= len(phonemes):
                continue
                
            # Calculate sample position
            sample_pos = int(marker.position * phoneme_duration)
            
            if marker.type == 'pause':
                # Insert pause
                pause_samples = int(marker.duration * sample_rate)
                silence = np.zeros(pause_samples)
                
                # Insert silence at position
                if sample_pos < len(modified_waveform):
                    modified_waveform = np.concatenate([
                        modified_waveform[:sample_pos],
                        silence,
                        modified_waveform[sample_pos:]
                    ])
            
            elif marker.type == 'stress':
                # Apply stress through amplitude and slight duration changes
                stress_region = int(0.1 * sample_rate)  # 100ms region
                start_pos = max(0, sample_pos - stress_region // 2)
                end_pos = min(len(modified_waveform), sample_pos + stress_region // 2)
                
                if end_pos > start_pos:
                    # Increase amplitude for stressed syllables
                    stress_gain = 1.0 + marker.strength * 0.2
                    modified_waveform[start_pos:end_pos] *= stress_gain
            
            elif marker.type in ['pitch_peak', 'pitch_final', 'pitch_set']:
                # Pitch modifications would require more complex processing
                # For now, apply subtle formant shifts to simulate pitch changes
                pitch_region = int(0.15 * sample_rate)  # 150ms region
                start_pos = max(0, sample_pos - pitch_region // 2)
                end_pos = min(len(modified_waveform), sample_pos + pitch_region // 2)
                
                if end_pos > start_pos:
                    # Simple pitch simulation through slight filtering
                    if marker.strength > 1.0:
                        # Higher pitch - slight high-frequency emphasis
                        from scipy import signal
                        b, a = signal.butter(1, 0.3, btype='highpass')
                        region = modified_waveform[start_pos:end_pos]
                        filtered = signal.filtfilt(b, a, region)
                        modified_waveform[start_pos:end_pos] = (
                            region * 0.8 + filtered * 0.2 * (marker.strength - 1.0)
                        )
        
        return modified_waveform
