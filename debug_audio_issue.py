#!/usr/bin/env python3
"""
Debug script to identify the audio quality issues in DJZ-Hawk
"""

import numpy as np
import matplotlib.pyplot as plt
from main import DJZHawk
import soundfile as sf

def analyze_waveform(waveform, title, filename=None):
    """Analyze and plot waveform characteristics"""
    print(f"\n=== {title} ===")
    print(f"Length: {len(waveform)} samples ({len(waveform)/22050:.2f} seconds)")
    print(f"Max amplitude: {np.max(np.abs(waveform)):.4f}")
    print(f"RMS level: {np.sqrt(np.mean(waveform**2)):.4f}")
    print(f"Dynamic range: {np.max(waveform) - np.min(waveform):.4f}")
    
    # Check for clipping
    clipped_samples = np.sum(np.abs(waveform) >= 0.99)
    print(f"Clipped samples: {clipped_samples} ({clipped_samples/len(waveform)*100:.2f}%)")
    
    # Check for silence
    silence_threshold = 0.001
    silent_samples = np.sum(np.abs(waveform) < silence_threshold)
    print(f"Silent samples: {silent_samples} ({silent_samples/len(waveform)*100:.2f}%)")
    
    # Save for analysis
    if filename:
        sf.write(filename, waveform, 22050)
        print(f"Saved to: {filename}")
    
    return waveform

def test_basic_synthesis():
    """Test basic synthesis components"""
    print("="*60)
    print("DEBUGGING DJZ-HAWK AUDIO ISSUES")
    print("="*60)
    
    hawk = DJZHawk()
    
    # Test 1: Simple text
    print("\n1. Testing simple text synthesis...")
    waveform1 = hawk.synthesize("Hello", save_file="debug_hello.wav")
    analyze_waveform(waveform1, "Simple Hello", "debug_hello_analysis.wav")
    
    # Test 2: Single phoneme
    print("\n2. Testing single phoneme...")
    from djz_hawk.src.diphone_synthesizer import DECtalkDiphoneSynthesizer
    synth = DECtalkDiphoneSynthesizer()
    
    # Generate a single diphone
    single_diphone = synth.diphone_db.get("AH_AH")
    if single_diphone:
        analyze_waveform(single_diphone.waveform, "Single Diphone AH_AH", "debug_single_diphone.wav")
    
    # Test 3: Raw phoneme generation
    print("\n3. Testing raw phoneme generation...")
    raw_waveform = synth._generate_phoneme_waveform("AH", "AH", 3307)  # 150ms at 22050Hz
    analyze_waveform(raw_waveform, "Raw Phoneme Generation", "debug_raw_phoneme.wav")
    
    # Test 4: Test without post-processing
    print("\n4. Testing synthesis without post-processing...")
    phonemes = ["AH"]
    diphones = synth._phonemes_to_diphones(phonemes)
    units = [synth._select_diphone_unit(d, synth._default_prosody()) for d in diphones]
    raw_concat = synth._concatenate_diphones(units, synth._default_prosody())
    analyze_waveform(raw_concat, "Raw Concatenation", "debug_raw_concat.wav")
    
    # Test 5: Test with post-processing
    print("\n5. Testing with post-processing...")
    processed = synth._apply_dectalk_characteristics(raw_concat)
    analyze_waveform(processed, "With Post-Processing", "debug_processed.wav")

def create_test_tone():
    """Create a simple test tone for comparison"""
    print("\n6. Creating test tone for comparison...")
    duration = 1.0  # 1 second
    sample_rate = 22050
    frequency = 440  # A4
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    test_tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    analyze_waveform(test_tone, "Test Tone (440Hz)", "debug_test_tone.wav")

if __name__ == "__main__":
    test_basic_synthesis()
    create_test_tone()
    
    print("\n" + "="*60)
    print("DEBUG ANALYSIS COMPLETE")
    print("Check the generated debug_*.wav files to identify the issue")
    print("="*60)
