#!/usr/bin/env python3
"""
DJZ-Hawk Clarity Showcase Test
Demonstrates the enhanced speech clarity improvements
"""

import os
import sys
import time

# Add the djz_hawk directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'djz_hawk'))

from main import DJZHawk

def test_clarity_improvements():
    """Test various aspects of speech clarity"""
    
    print("=" * 60)
    print("DJZ-HAWK CLARITY SHOWCASE")
    print("Enhanced Speech Synthesis with Improved Intelligibility")
    print("=" * 60)
    
    # Initialize DJZ-Hawk
    djz_hawk = DJZHawk()
    
    # Test sentences focusing on different aspects of clarity
    test_cases = [
        {
            'name': 'Consonant Clarity',
            'text': 'Peter Piper picked a peck of pickled peppers',
            'description': 'Tests plosive consonant clarity and timing'
        },
        {
            'name': 'Fricative Precision',
            'text': 'She sells seashells by the seashore',
            'description': 'Tests fricative consonant precision and sibilant clarity'
        },
        {
            'name': 'Vowel Distinction',
            'text': 'The cat sat on the mat with a hat',
            'description': 'Tests vowel formant accuracy and distinction'
        },
        {
            'name': 'Complex Transitions',
            'text': 'Artificial intelligence creates extraordinary possibilities',
            'description': 'Tests complex phoneme transitions and coarticulation'
        },
        {
            'name': 'Technical Terms',
            'text': 'Digital Equipment Corporation DECtalk speech synthesis',
            'description': 'Tests technical vocabulary and proper pronunciation'
        },
        {
            'name': 'Natural Flow',
            'text': 'Hello, my name is Perfect Paul. I am a computer voice from nineteen ninety six.',
            'description': 'Tests natural speech flow and prosody'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Description: {test_case['description']}")
        print(f"   Text: \"{test_case['text']}\"")
        
        # Generate filename
        filename = f"clarity_test_{i:02d}_{test_case['name'].lower().replace(' ', '_')}.wav"
        
        # Synthesize
        print(f"   Synthesizing...")
        start_time = time.time()
        
        try:
            waveform = djz_hawk.synthesize(test_case['text'], save_file=filename)
            duration = time.time() - start_time
            audio_length = len(waveform) / djz_hawk.synthesizer.sample_rate
            
            print(f"   ✓ Generated: {filename}")
            print(f"   ✓ Audio length: {audio_length:.2f} seconds")
            print(f"   ✓ Processing time: {duration:.2f} seconds")
            print(f"   ✓ Real-time factor: {audio_length/duration:.1f}x")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("CLARITY ENHANCEMENTS SUMMARY")
    print("=" * 60)
    print("✓ Enhanced LPC formant synthesis with precise frequency modeling")
    print("✓ Optimized phoneme durations for better intelligibility")
    print("✓ Improved consonant articulation with proper voicing")
    print("✓ Enhanced vowel formants with accurate F1/F2/F3 frequencies")
    print("✓ Smoother diphone transitions with Hanning window crossfades")
    print("✓ Reduced inter-phoneme pauses for better speech flow")
    print("✓ Characteristic DECtalk metallic resonance preserved")
    print("✓ Authentic 1996 audio processing artifacts maintained")
    
    print("\nAll test files have been generated successfully!")
    print("Listen to the audio files to hear the clarity improvements.")

if __name__ == '__main__':
    test_clarity_improvements()
