"""
Phoneme Engine for DJZ-Hawk
ARPABET phoneme processing with context rules for DECtalk 4.2CD recreation
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PhonemeContext:
    """Context information for phoneme processing"""
    preceding_phoneme: Optional[str]
    following_phoneme: Optional[str]
    word_position: str  # 'initial', 'medial', 'final'
    syllable_position: str  # 'onset', 'nucleus', 'coda'
    stress_level: int  # 0=unstressed, 1=primary, 2=secondary

class DECtalkPhonemeEngine:
    """
    Phoneme engine implementing ARPABET with DECtalk 4.2CD context rules
    Handles phoneme-to-phoneme conversion and contextual modifications
    """
    
    def __init__(self):
        self.arpabet_map = self._load_arpabet_mapping()
        self.context_rules = self._load_context_rules()
        self.coarticulation_rules = self._load_coarticulation_rules()
        self.dectalk_substitutions = self._load_dectalk_substitutions()
        
    def _load_arpabet_mapping(self) -> Dict[str, str]:
        """Load ARPABET phoneme mappings"""
        return {
            # Vowels (monophthongs)
            'AA': 'aa',  # father, hot
            'AE': 'ae',  # cat, bat
            'AH': 'ah',  # but, cut
            'AO': 'ao',  # caught, law
            'AW': 'aw',  # how, now
            'AY': 'ay',  # my, eye
            'EH': 'eh',  # bet, red
            'ER': 'er',  # bird, hurt
            'EY': 'ey',  # bay, say
            'IH': 'ih',  # bit, hit
            'IY': 'iy',  # beat, heat
            'OW': 'ow',  # boat, note
            'OY': 'oy',  # boy, toy
            'UH': 'uh',  # book, put
            'UW': 'uw',  # boot, food
            
            # Consonants (stops)
            'B': 'b',    # bat
            'D': 'd',    # dog
            'G': 'g',    # go
            'P': 'p',    # pat
            'T': 't',    # top
            'K': 'k',    # cat
            
            # Fricatives
            'DH': 'dh',  # this
            'F': 'f',    # fish
            'HH': 'hh',  # house
            'S': 's',    # see
            'SH': 'sh',  # she
            'TH': 'th',  # think
            'V': 'v',    # very
            'Z': 'z',    # zoo
            'ZH': 'zh',  # measure
            
            # Nasals
            'M': 'm',    # man
            'N': 'n',    # no
            'NG': 'ng',  # sing
            
            # Liquids
            'L': 'l',    # left
            'R': 'r',    # red
            
            # Glides
            'W': 'w',    # we
            'Y': 'y',    # yes
            
            # Affricates
            'CH': 'ch',  # chair
            'JH': 'jh',  # just
        }
    
    def _load_context_rules(self) -> Dict[str, List[Dict]]:
        """Load phoneme context modification rules"""
        return {
            # Vowel modifications in different contexts
            'vowel_reduction': [
                {
                    'condition': 'unstressed_syllable',
                    'target_phonemes': ['AE', 'EH', 'IH', 'AH'],
                    'replacement': 'AH',  # Schwa reduction
                    'probability': 0.7
                },
                {
                    'condition': 'word_final_unstressed',
                    'target_phonemes': ['IY', 'EY'],
                    'replacement': 'IH',
                    'probability': 0.5
                }
            ],
            
            # Consonant modifications
            'consonant_weakening': [
                {
                    'condition': 'intervocalic',
                    'target_phonemes': ['T', 'D'],
                    'replacement': 'DX',  # Flap
                    'probability': 0.8
                },
                {
                    'condition': 'word_final',
                    'target_phonemes': ['T', 'D', 'K', 'G', 'P', 'B'],
                    'modification': 'unreleased',
                    'probability': 0.6
                }
            ],
            
            # DECtalk-specific rules
            'dectalk_assimilation': [
                {
                    'condition': 'before_dental',
                    'target_phonemes': ['T', 'D', 'N', 'L', 'S', 'Z'],
                    'modification': 'dental',
                    'probability': 0.3  # DECtalk's characteristic feature
                }
            ]
        }
    
    def _load_coarticulation_rules(self) -> Dict[str, List[Dict]]:
        """Load coarticulation rules for natural speech flow"""
        return {
            'anticipatory': [
                # Vowel anticipation before rounded consonants
                {
                    'trigger': ['W', 'UW', 'OW'],
                    'target': 'preceding_vowel',
                    'effect': 'lip_rounding',
                    'strength': 0.3
                },
                # Nasal anticipation
                {
                    'trigger': ['M', 'N', 'NG'],
                    'target': 'preceding_vowel',
                    'effect': 'nasalization',
                    'strength': 0.4
                }
            ],
            
            'carryover': [
                # Consonant voicing carryover
                {
                    'trigger': ['B', 'D', 'G', 'V', 'Z', 'ZH', 'DH', 'JH'],
                    'target': 'following_consonant',
                    'effect': 'voicing',
                    'strength': 0.2
                },
                # Fricative carryover
                {
                    'trigger': ['S', 'SH', 'F', 'TH'],
                    'target': 'following_vowel',
                    'effect': 'breathiness',
                    'strength': 0.15
                }
            ]
        }
    
    def _load_dectalk_substitutions(self) -> Dict[str, str]:
        """Load DECtalk-specific phoneme substitutions"""
        return {
            # DECtalk's characteristic pronunciations
            'the': 'DH AH',      # Reduced form
            'and': 'AH N D',     # Reduced vowel
            'of': 'AH V',        # Weak form
            'to': 'T AH',        # Reduced form
            'for': 'F ER',       # DECtalk's characteristic pronunciation
            'with': 'W IH TH',   # Clear pronunciation
            'from': 'F R AH M',  # Reduced vowel
            'have': 'HH AE V',   # Full vowel (DECtalk didn't reduce this)
            'been': 'B IH N',    # DECtalk's pronunciation
            'were': 'W ER',      # R-colored vowel
            'said': 'S EH D',    # DECtalk's vowel choice
        }
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to ARPABET phonemes using DECtalk rules
        """
        # 1. Normalize text
        normalized_text = self._normalize_text(text)
        
        # 2. Split into words
        words = normalized_text.split()
        
        # 3. Convert each word to phonemes
        all_phonemes = []
        for word in words:
            word_phonemes = self._word_to_phonemes(word)
            all_phonemes.extend(word_phonemes)
            
            # Add word boundary marker
            if word != words[-1]:  # Not the last word
                all_phonemes.append('_')  # Word boundary
        
        # 4. Apply context rules
        processed_phonemes = self._apply_context_rules(all_phonemes)
        
        # 5. Apply DECtalk-specific modifications
        final_phonemes = self._apply_dectalk_modifications(processed_phonemes)
        
        return final_phonemes
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for phoneme conversion"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Handle contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'d": " would",
            "'m": " am",
            "'s": " is"  # Simplified - could be possessive
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _word_to_phonemes(self, word: str) -> List[str]:
        """Convert a single word to phonemes"""
        word = word.strip().lower()
        
        # Check for direct substitutions first
        if word in self.dectalk_substitutions:
            return self.dectalk_substitutions[word].split()
        
        # Use simplified grapheme-to-phoneme rules
        # This is a basic implementation - a full system would use a dictionary
        phonemes = self._grapheme_to_phoneme(word)
        
        return phonemes
    
    def _grapheme_to_phoneme(self, word: str) -> List[str]:
        """
        Basic grapheme-to-phoneme conversion
        This is simplified - a full implementation would use CMU dict or similar
        """
        phonemes = []
        i = 0
        
        while i < len(word):
            # Handle digraphs and trigraphs first
            if i < len(word) - 2:
                trigraph = word[i:i+3]
                if trigraph in self._get_trigraph_rules():
                    phonemes.extend(self._get_trigraph_rules()[trigraph])
                    i += 3
                    continue
            
            if i < len(word) - 1:
                digraph = word[i:i+2]
                if digraph in self._get_digraph_rules():
                    phonemes.extend(self._get_digraph_rules()[digraph])
                    i += 2
                    continue
            
            # Single character rules
            char = word[i]
            if char in self._get_single_char_rules():
                phoneme = self._get_single_char_rules()[char]
                if phoneme:  # Skip silent letters
                    phonemes.append(phoneme)
            
            i += 1
        
        return phonemes
    
    def _get_trigraph_rules(self) -> Dict[str, List[str]]:
        """Get trigraph pronunciation rules"""
        return {
            'sch': ['S', 'K'],
            'tch': ['CH'],
            'dge': ['JH'],
            'igh': ['AY'],
            'ough': ['AH', 'F'],  # rough
            'augh': ['AO', 'F'],  # laugh
        }
    
    def _get_digraph_rules(self) -> Dict[str, List[str]]:
        """Get digraph pronunciation rules"""
        return {
            'th': ['TH'],    # Default to voiceless
            'sh': ['SH'],
            'ch': ['CH'],
            'ph': ['F'],
            'gh': ['G'],     # ghost
            'ng': ['NG'],
            'ck': ['K'],
            'qu': ['K', 'W'],
            'wh': ['W'],     # Simplified
            'kn': ['N'],     # Silent k
            'wr': ['R'],     # Silent w
            'mb': ['M'],     # Silent b (lamb)
            'bt': ['T'],     # Silent b (debt)
            'mn': ['M'],     # Silent n (hymn)
            'ps': ['S'],     # Silent p (psalm)
            'pt': ['T'],     # Silent p (pterodactyl)
            'rh': ['R'],     # Silent h (rhyme)
            'sc': ['S'],     # Silent c (scene)
            'sw': ['S', 'W'],
            'tw': ['T', 'W'],
            'dw': ['D', 'W'],
            'gw': ['G', 'W'],
            'oo': ['UW'],    # Default to long
            'ee': ['IY'],
            'ea': ['IY'],    # Default
            'ai': ['EY'],
            'ay': ['EY'],
            'oa': ['OW'],
            'ow': ['OW'],    # Default to long
            'ou': ['AW'],    # Default
            'oi': ['OY'],
            'oy': ['OY'],
            'au': ['AO'],
            'aw': ['AO'],
            'ie': ['IY'],    # Default
            'ei': ['EY'],    # Default
            'ue': ['UW'],
            'ui': ['UW'],
            'ar': ['AA', 'R'],
            'er': ['ER'],
            'ir': ['ER'],
            'or': ['AO', 'R'],
            'ur': ['ER'],
            'yr': ['ER'],
        }
    
    def _get_single_char_rules(self) -> Dict[str, str]:
        """Get single character pronunciation rules"""
        return {
            'a': 'AE',   # Default short
            'b': 'B',
            'c': 'K',    # Default hard
            'd': 'D',
            'e': 'EH',   # Default short
            'f': 'F',
            'g': 'G',    # Default hard
            'h': 'HH',
            'i': 'IH',   # Default short
            'j': 'JH',
            'k': 'K',
            'l': 'L',
            'm': 'M',
            'n': 'N',
            'o': 'AH',   # Default short
            'p': 'P',
            'q': 'K',
            'r': 'R',
            's': 'S',
            't': 'T',
            'u': 'AH',   # Default short
            'v': 'V',
            'w': 'W',
            'x': 'K',    # Simplified
            'y': 'Y',    # Default consonant
            'z': 'Z',
        }
    
    def _apply_context_rules(self, phonemes: List[str]) -> List[str]:
        """Apply context-dependent phoneme modifications"""
        modified_phonemes = phonemes.copy()
        
        for i, phoneme in enumerate(phonemes):
            if phoneme == '_':  # Skip word boundaries
                continue
            
            # Get context
            context = self._get_phoneme_context(phonemes, i)
            
            # Apply vowel reduction rules
            if phoneme in ['AE', 'EH', 'IH', 'AH'] and context.stress_level == 0:
                # Unstressed vowels tend to reduce to schwa
                if np.random.random() < 0.3:  # 30% chance
                    modified_phonemes[i] = 'AH'
            
            # Apply consonant modifications
            if phoneme in ['T', 'D'] and self._is_intervocalic(phonemes, i):
                # Intervocalic flapping
                if np.random.random() < 0.4:  # 40% chance
                    modified_phonemes[i] = 'DX'  # Flap
        
        return modified_phonemes
    
    def _get_phoneme_context(self, phonemes: List[str], position: int) -> PhonemeContext:
        """Get context information for a phoneme"""
        preceding = phonemes[position - 1] if position > 0 else None
        following = phonemes[position + 1] if position < len(phonemes) - 1 else None
        
        # Simplified context analysis
        word_position = 'medial'
        if position == 0 or (position > 0 and phonemes[position - 1] == '_'):
            word_position = 'initial'
        elif position == len(phonemes) - 1 or (position < len(phonemes) - 1 and phonemes[position + 1] == '_'):
            word_position = 'final'
        
        # Simplified syllable position
        syllable_position = 'nucleus'  # Default
        if phonemes[position] in ['B', 'D', 'G', 'P', 'T', 'K', 'F', 'V', 'S', 'Z', 'SH', 'ZH', 'TH', 'DH', 'M', 'N', 'NG', 'L', 'R', 'W', 'Y', 'HH', 'CH', 'JH']:
            syllable_position = 'onset' if word_position == 'initial' else 'coda'
        
        return PhonemeContext(
            preceding_phoneme=preceding,
            following_phoneme=following,
            word_position=word_position,
            syllable_position=syllable_position,
            stress_level=1  # Simplified - assume primary stress
        )
    
    def _is_intervocalic(self, phonemes: List[str], position: int) -> bool:
        """Check if phoneme is between vowels"""
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
        
        if position == 0 or position >= len(phonemes) - 1:
            return False
        
        prev_phoneme = phonemes[position - 1]
        next_phoneme = phonemes[position + 1]
        
        return prev_phoneme in vowels and next_phoneme in vowels
    
    def _apply_dectalk_modifications(self, phonemes: List[str]) -> List[str]:
        """Apply DECtalk-specific phoneme modifications"""
        modified_phonemes = []
        
        for i, phoneme in enumerate(phonemes):
            if phoneme == '_':  # Skip word boundaries
                continue
            
            # Apply DECtalk's characteristic alveolar->dental assimilation
            if phoneme in ['T', 'D', 'N', 'L', 'S', 'Z']:
                # Check if followed by dental context
                if i < len(phonemes) - 1:
                    next_phoneme = phonemes[i + 1]
                    if next_phoneme in ['TH', 'DH']:
                        # Apply dental assimilation with some probability
                        if np.random.random() < 0.25:  # 25% chance
                            phoneme = phoneme.lower() + '_dental'
            
            # Convert to DECtalk internal format
            if phoneme in self.arpabet_map:
                modified_phonemes.append(self.arpabet_map[phoneme])
            else:
                modified_phonemes.append(phoneme.lower())
        
        return modified_phonemes
    
    def phonemes_to_diphones(self, phonemes: List[str]) -> List[str]:
        """Convert phoneme sequence to diphone sequence"""
        if len(phonemes) < 2:
            return phonemes
        
        diphones = []
        
        # Add initial silence-phoneme diphone
        diphones.append(f"sil_{phonemes[0]}")
        
        # Add phoneme-phoneme diphones
        for i in range(len(phonemes) - 1):
            diphone = f"{phonemes[i]}_{phonemes[i + 1]}"
            diphones.append(diphone)
        
        # Add final phoneme-silence diphone
        diphones.append(f"{phonemes[-1]}_sil")
        
        return diphones
    
    def analyze_phoneme_features(self, phoneme: str) -> Dict[str, any]:
        """Analyze phonetic features of a phoneme"""
        features = {
            'type': 'unknown',
            'voicing': 'unknown',
            'place': 'unknown',
            'manner': 'unknown',
            'height': 'unknown',
            'backness': 'unknown',
            'rounding': 'unknown'
        }
        
        # Vowel features
        vowel_features = {
            'aa': {'type': 'vowel', 'height': 'low', 'backness': 'back', 'rounding': 'unrounded'},
            'ae': {'type': 'vowel', 'height': 'low', 'backness': 'front', 'rounding': 'unrounded'},
            'ah': {'type': 'vowel', 'height': 'mid', 'backness': 'central', 'rounding': 'unrounded'},
            'ao': {'type': 'vowel', 'height': 'low', 'backness': 'back', 'rounding': 'rounded'},
            'eh': {'type': 'vowel', 'height': 'mid', 'backness': 'front', 'rounding': 'unrounded'},
            'er': {'type': 'vowel', 'height': 'mid', 'backness': 'central', 'rounding': 'unrounded'},
            'ih': {'type': 'vowel', 'height': 'high', 'backness': 'front', 'rounding': 'unrounded'},
            'iy': {'type': 'vowel', 'height': 'high', 'backness': 'front', 'rounding': 'unrounded'},
            'ow': {'type': 'vowel', 'height': 'mid', 'backness': 'back', 'rounding': 'rounded'},
            'uh': {'type': 'vowel', 'height': 'high', 'backness': 'back', 'rounding': 'rounded'},
            'uw': {'type': 'vowel', 'height': 'high', 'backness': 'back', 'rounding': 'rounded'},
        }
        
        # Consonant features
        consonant_features = {
            'b': {'type': 'consonant', 'voicing': 'voiced', 'place': 'bilabial', 'manner': 'stop'},
            'd': {'type': 'consonant', 'voicing': 'voiced', 'place': 'alveolar', 'manner': 'stop'},
            'g': {'type': 'consonant', 'voicing': 'voiced', 'place': 'velar', 'manner': 'stop'},
            'p': {'type': 'consonant', 'voicing': 'voiceless', 'place': 'bilabial', 'manner': 'stop'},
            't': {'type': 'consonant', 'voicing': 'voiceless', 'place': 'alveolar', 'manner': 'stop'},
            'k': {'type': 'consonant', 'voicing': 'voiceless', 'place': 'velar', 'manner': 'stop'},
            'f': {'type': 'consonant', 'voicing': 'voiceless', 'place': 'labiodental', 'manner': 'fricative'},
            'v': {'type': 'consonant', 'voicing': 'voiced', 'place': 'labiodental', 'manner': 'fricative'},
            's': {'type': 'consonant', 'voicing': 'voiceless', 'place': 'alveolar', 'manner': 'fricative'},
            'z': {'type': 'consonant', 'voicing': 'voiced', 'place': 'alveolar', 'manner': 'fricative'},
            'sh': {'type': 'consonant', 'voicing': 'voiceless', 'place': 'postalveolar', 'manner': 'fricative'},
            'zh': {'type': 'consonant', 'voicing': 'voiced', 'place': 'postalveolar', 'manner': 'fricative'},
            'th': {'type': 'consonant', 'voicing': 'voiceless', 'place': 'dental', 'manner': 'fricative'},
            'dh': {'type': 'consonant', 'voicing': 'voiced', 'place': 'dental', 'manner': 'fricative'},
            'hh': {'type': 'consonant', 'voicing': 'voiceless', 'place': 'glottal', 'manner': 'fricative'},
            'm': {'type': 'consonant', 'voicing': 'voiced', 'place': 'bilabial', 'manner': 'nasal'},
            'n': {'type': 'consonant', 'voicing': 'voiced', 'place': 'alveolar', 'manner': 'nasal'},
            'ng': {'type': 'consonant', 'voicing': 'voiced', 'place': 'velar', 'manner': 'nasal'},
            'l': {'type': 'consonant', 'voicing': 'voiced', 'place': 'alveolar', 'manner': 'liquid'},
            'r': {'type': 'consonant', 'voicing': 'voiced', 'place': 'alveolar', 'manner': 'liquid'},
            'w': {'type': 'consonant', 'voicing': 'voiced', 'place': 'bilabial', 'manner': 'glide'},
            'y': {'type': 'consonant', 'voicing': 'voiced', 'place': 'palatal', 'manner': 'glide'},
            'ch': {'type': 'consonant', 'voicing': 'voiceless', 'place': 'postalveolar', 'manner': 'affricate'},
            'jh': {'type': 'consonant', 'voicing': 'voiced', 'place': 'postalveolar', 'manner': 'affricate'},
        }
        
        # Look up features
        if phoneme in vowel_features:
            features.update(vowel_features[phoneme])
        elif phoneme in consonant_features:
            features.update(consonant_features[phoneme])
        
        return features

# Import numpy for random operations
import numpy as np
