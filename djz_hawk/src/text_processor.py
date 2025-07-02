"""
Text Processor for DJZ-Hawk
Implements 1996-era text normalization with period-appropriate rules
"""

import re
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class TextContext:
    """Context information for text processing"""
    sentence_position: str  # 'start', 'middle', 'end'
    punctuation: str
    emphasis_level: int
    abbreviation_context: bool

class DECtalk96TextProcessor:
    """
    Faithful recreation of DECtalk 4.2CD text processing engine
    Implements 1996-era abbreviation expansion and normalization rules
    """
    
    def __init__(self):
        self.abbreviations = self._load_1996_abbreviations()
        self.number_rules = self._load_number_rules()
        self.punctuation_prosody = self._load_punctuation_mapping()
        
    def _load_1996_abbreviations(self) -> dict:
        """Load period-appropriate abbreviation expansions"""
        return {
            'Dr.': 'Doctor',
            'Mr.': 'Mister', 
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'St.': self._context_aware_saint_street,
            'Ave.': 'Avenue',
            'Blvd.': 'Boulevard',
            'Co.': 'Company',
            'Corp.': 'Corporation',
            'Inc.': 'Incorporated',
            'Ltd.': 'Limited',
            'ft.': 'feet',
            'in.': 'inches',
            'lb.': 'pound',
            'oz.': 'ounce',
            'vs.': 'versus',
            'etc.': 'etcetera',
            'e.g.': 'for example',
            'i.e.': 'that is',
            # 1996-specific tech abbreviations
            'RAM': 'random access memory',
            'CPU': 'central processing unit',
            'ISA': 'industry standard architecture',
            'PCI': 'peripheral component interconnect',
            'SCSI': 'scuzzy',
            'IDE': 'integrated drive electronics'
        }
    
    def _load_number_rules(self) -> dict:
        """Load number pronunciation rules"""
        return {
            'ordinals': {
                '1st': 'first', '2nd': 'second', '3rd': 'third',
                '4th': 'fourth', '5th': 'fifth', '6th': 'sixth',
                '7th': 'seventh', '8th': 'eighth', '9th': 'ninth',
                '10th': 'tenth', '11th': 'eleventh', '12th': 'twelfth'
            },
            'cardinals': {
                '0': 'zero', '1': 'one', '2': 'two', '3': 'three',
                '4': 'four', '5': 'five', '6': 'six', '7': 'seven',
                '8': 'eight', '9': 'nine', '10': 'ten'
            }
        }
    
    def _load_punctuation_mapping(self) -> dict:
        """Load punctuation to prosody mapping"""
        return {
            '.': {'pause': 0.5, 'pitch_fall': True},
            '!': {'pause': 0.4, 'pitch_fall': True, 'emphasis': True},
            '?': {'pause': 0.4, 'pitch_rise': True},
            ',': {'pause': 0.25, 'pitch_slight_fall': True},
            ';': {'pause': 0.35, 'pitch_fall': True},
            ':': {'pause': 0.3, 'pitch_rise': True}
        }
    
    def _context_aware_saint_street(self, context: str) -> str:
        """Context-aware St. expansion (Saint vs Street)"""
        # Look for numeric context for street
        if re.search(r'\d+.*St\.', context):
            return 'Street'
        # Look for name context for Saint
        elif re.search(r'St\.\s+[A-Z][a-z]+', context):
            return 'Saint'
        else:
            return 'Street'  # Default to street in ambiguous cases
    
    def process_text(self, text: str) -> List[Tuple[str, TextContext]]:
        """
        Main text processing pipeline
        Returns list of (processed_text, context) tuples
        """
        # 1. Basic cleanup and normalization
        text = self._normalize_whitespace(text)
        text = self._handle_contractions(text)
        
        # 2. Sentence segmentation
        sentences = self._segment_sentences(text)
        
        # 3. Process each sentence
        processed_segments = []
        for i, sentence in enumerate(sentences):
            context = TextContext(
                sentence_position='start' if i == 0 else 'end' if i == len(sentences)-1 else 'middle',
                punctuation=self._extract_punctuation(sentence),
                emphasis_level=self._calculate_emphasis(sentence),
                abbreviation_context=self._has_abbreviations(sentence)
            )
            
            # Apply 1996-era processing rules
            processed = self._expand_abbreviations(sentence)
            processed = self._process_numbers(processed)
            processed = self._handle_symbols(processed)
            processed = self._apply_pronunciation_rules(processed)
            
            processed_segments.append((processed, context))
        
        return processed_segments
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        return re.sub(r'\s+', ' ', text).strip()
    
    def _handle_contractions(self, text: str) -> str:
        """Handle contractions with 1996-era rules"""
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'d": " would",
            "'m": " am",
            "'s": " is"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences"""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_punctuation(self, sentence: str) -> str:
        """Extract punctuation from sentence"""
        punctuation = re.findall(r'[.!?,:;]', sentence)
        return ''.join(punctuation)
    
    def _calculate_emphasis(self, sentence: str) -> int:
        """Calculate emphasis level for sentence"""
        emphasis_markers = ['!', 'very', 'really', 'extremely']
        emphasis_count = sum(1 for marker in emphasis_markers if marker in sentence.lower())
        return min(emphasis_count, 3)  # Cap at level 3
    
    def _has_abbreviations(self, sentence: str) -> bool:
        """Check if sentence contains abbreviations"""
        for abbrev in self.abbreviations.keys():
            if abbrev in sentence:
                return True
        return False
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations with 1996-era rules"""
        for abbrev, expansion in self.abbreviations.items():
            if callable(expansion):
                # Handle context-aware expansions
                matches = re.finditer(re.escape(abbrev), text)
                for match in reversed(list(matches)):
                    start, end = match.span()
                    context = text[max(0, start-20):min(len(text), end+20)]
                    expanded = expansion(context)
                    text = text[:start] + expanded + text[end:]
            else:
                text = text.replace(abbrev, expansion)
        
        return text
    
    def _process_numbers(self, text: str) -> str:
        """Process numbers with 1996-era rules"""
        # Ordinals (1st, 2nd, 3rd, etc.)
        for ordinal, word in self.number_rules['ordinals'].items():
            text = re.sub(r'\b' + re.escape(ordinal) + r'\b', word, text, flags=re.IGNORECASE)
        
        # Years (handle 1990s appropriately)
        text = re.sub(r'\b19(\d{2})\b', self._pronounce_year, text)
        
        # Large numbers with commas
        text = re.sub(r'\b(\d{1,3}(?:,\d{3})+)\b', self._pronounce_large_number, text)
        
        # Decimal numbers
        text = re.sub(r'\b(\d+)\.(\d+)\b', self._pronounce_decimal, text)
        
        # Simple digits
        for digit, word in self.number_rules['cardinals'].items():
            text = re.sub(r'\b' + digit + r'\b', word, text)
        
        return text
    
    def _pronounce_year(self, match) -> str:
        """Pronounce years in 1990s style"""
        year_suffix = match.group(1)
        if year_suffix in ['90', '91', '92', '93', '94', '95', '96', '97', '98', '99']:
            return f"nineteen {self._number_to_words(int(year_suffix))}"
        return f"nineteen {self._number_to_words(int(year_suffix))}"
    
    def _pronounce_large_number(self, match) -> str:
        """Pronounce large numbers with commas"""
        number_str = match.group(1).replace(',', '')
        try:
            number = int(number_str)
            return self._number_to_words(number)
        except ValueError:
            return match.group(1)
    
    def _pronounce_decimal(self, match) -> str:
        """Pronounce decimal numbers"""
        whole_part = match.group(1)
        decimal_part = match.group(2)
        
        whole_words = self._number_to_words(int(whole_part))
        decimal_words = ' '.join(self._number_to_words(int(d)) for d in decimal_part)
        
        return f"{whole_words} point {decimal_words}"
    
    def _number_to_words(self, number: int) -> str:
        """Convert number to words"""
        if number == 0:
            return "zero"
        
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen"]
        
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if number < 20:
            return ones[number]
        elif number < 100:
            return tens[number // 10] + ("" if number % 10 == 0 else " " + ones[number % 10])
        elif number < 1000:
            return ones[number // 100] + " hundred" + ("" if number % 100 == 0 else " " + self._number_to_words(number % 100))
        elif number < 1000000:
            return self._number_to_words(number // 1000) + " thousand" + ("" if number % 1000 == 0 else " " + self._number_to_words(number % 1000))
        else:
            return str(number)  # Fallback for very large numbers
    
    def _handle_symbols(self, text: str) -> str:
        """Handle symbols and special characters"""
        symbol_map = {
            '&': 'and',
            '@': 'at',
            '#': 'number',
            '$': 'dollar',
            '%': 'percent',
            '+': 'plus',
            '=': 'equals',
            '<': 'less than',
            '>': 'greater than',
            '/': 'slash',
            '\\': 'backslash',
            '*': 'asterisk',
            '^': 'caret'
        }
        
        for symbol, word in symbol_map.items():
            text = text.replace(symbol, f' {word} ')
        
        return text
    
    def _apply_pronunciation_rules(self, text: str) -> str:
        """Apply DECtalk-specific pronunciation rules"""
        # Common mispronunciations that were characteristic of DECtalk
        replacements = {
            r'\bthe\b': 'thuh',     # Reduced vowel in unstressed position
            r'\band\b': 'uhnd',     # Typical reduction
            r'\bof\b': 'uhv',       # Weak form
            r'\bto\b': 'tuh',       # Reduced form
            r'\bfor\b': 'fehr',     # DECtalk's characteristic pronunciation
            r'\bwith\b': 'wihth',   # Slight modification
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert processed text to phonemes
        This is a simplified version - delegates to phoneme engine
        """
        # This method serves as a bridge to the phoneme engine
        # In a full implementation, this would be more sophisticated
        words = text.split()
        phonemes = []
        
        for word in words:
            # Simple mapping - in reality this would use the phoneme engine
            word_phonemes = self._simple_word_to_phonemes(word)
            phonemes.extend(word_phonemes)
        
        return phonemes
    
    def _simple_word_to_phonemes(self, word: str) -> List[str]:
        """
        Simple word to phoneme conversion
        This is a placeholder - real implementation would use phoneme engine
        """
        word = word.lower().strip()
        
        # Simple word-level phoneme mapping for common words
        word_phonemes = {
            'hello': ['HH', 'EH', 'L', 'OW'],
            'hi': ['HH', 'AY'],
            'the': ['DH', 'AH'],
            'this': ['DH', 'IH', 'S'],
            'is': ['IH', 'Z'],
            'a': ['AH'],
            'an': ['AE', 'N'],
            'and': ['AE', 'N', 'D'],
            'to': ['T', 'UW'],
            'of': ['AH', 'V'],
            'in': ['IH', 'N'],
            'for': ['F', 'AO', 'R'],
            'with': ['W', 'IH', 'TH'],
            'on': ['AA', 'N'],
            'at': ['AE', 'T'],
            'be': ['B', 'IY'],
            'have': ['HH', 'AE', 'V'],
            'it': ['IH', 'T'],
            'that': ['DH', 'AE', 'T'],
            'not': ['N', 'AA', 'T'],
            'you': ['Y', 'UW'],
            'all': ['AO', 'L'],
            'can': ['K', 'AE', 'N'],
            'had': ['HH', 'AE', 'D'],
            'her': ['HH', 'ER'],
            'was': ['W', 'AA', 'Z'],
            'one': ['W', 'AH', 'N'],
            'our': ['AW', 'R'],
            'out': ['AW', 'T'],
            'day': ['D', 'EY'],
            'get': ['G', 'EH', 'T'],
            'has': ['HH', 'AE', 'Z'],
            'him': ['HH', 'IH', 'M'],
            'his': ['HH', 'IH', 'Z'],
            'how': ['HH', 'AW'],
            'man': ['M', 'AE', 'N'],
            'new': ['N', 'UW'],
            'now': ['N', 'AW'],
            'old': ['OW', 'L', 'D'],
            'see': ['S', 'IY'],
            'two': ['T', 'UW'],
            'way': ['W', 'EY'],
            'who': ['HH', 'UW'],
            'boy': ['B', 'OY'],
            'did': ['D', 'IH', 'D'],
            'its': ['IH', 'T', 'S'],
            'let': ['L', 'EH', 'T'],
            'put': ['P', 'UH', 'T'],
            'say': ['S', 'EY'],
            'she': ['SH', 'IY'],
            'too': ['T', 'UW'],
            'use': ['Y', 'UW', 'Z'],
            'should': ['SH', 'UH', 'D'],
            'much': ['M', 'AH', 'CH'],
            'better': ['B', 'EH', 'T', 'ER'],
            'clearer': ['K', 'L', 'IH', 'R', 'ER'],
            'sound': ['S', 'AW', 'N', 'D'],
            'now': ['N', 'AW'],
            'synthesis': ['S', 'IH', 'N', 'TH', 'AH', 'S', 'IH', 'S'],
            'speech': ['S', 'P', 'IY', 'CH'],
            'voice': ['V', 'OY', 'S'],
            'test': ['T', 'EH', 'S', 'T'],
            'testing': ['T', 'EH', 'S', 'T', 'IH', 'NG'],
            'hawk': ['HH', 'AO', 'K'],
            'djz': ['D', 'IY', 'JH', 'EY', 'Z', 'IY'],
            'paul': ['P', 'AO', 'L'],
            'perfect': ['P', 'ER', 'F', 'EH', 'K', 'T'],
            'beautiful': ['B', 'Y', 'UW', 'T', 'AH', 'F', 'AH', 'L'],
            'betty': ['B', 'EH', 'T', 'IY'],
            'huge': ['HH', 'Y', 'UW', 'JH'],
            'harry': ['HH', 'EH', 'R', 'IY'],
            'kit': ['K', 'IH', 'T'],
            'kid': ['K', 'IH', 'D'],
            'frank': ['F', 'R', 'AE', 'NG', 'K'],
            'rita': ['R', 'IY', 'T', 'AH'],
            'ursula': ['ER', 'S', 'AH', 'L', 'AH'],
            'val': ['V', 'AE', 'L'],
            'rough': ['R', 'AH', 'F']
        }
        
        if word in word_phonemes:
            return word_phonemes[word]
        
        # Fallback: simple character-to-phoneme mapping for unknown words
        char_phonemes = {
            'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AH', 'u': 'UH',
            'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
            'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
            'n': 'N', 'p': 'P', 'r': 'R', 's': 'S', 't': 'T',
            'v': 'V', 'w': 'W', 'y': 'Y', 'z': 'Z'
        }
        
        phonemes = []
        for char in word:
            if char in char_phonemes:
                phonemes.append(char_phonemes[char])
        
        return phonemes if phonemes else ['SIL']
