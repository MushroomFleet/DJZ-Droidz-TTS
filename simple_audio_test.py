#!/usr/bin/env python3
"""
Simple audio test to isolate the static pulse issue
"""

import numpy as np
import soundfile as sf
from djz_hawk.src.diphone_synthesizer import DECtalkDiphoneSynthesizer

def test_raw_synthesis():
    """Test the raw synthesis without all the processing"""
    print("Testing raw synthesis components...")
    
    synth = DECtalkDiphoneSynthesizer()
    
    # Test 1: Generate a single vowel sound
    print("\n1. Testing single vowel generation...")
    samples = int(0.5 * 22050)  # 0.5 seconds
    vowel_waveform = synth._generate_vowel_waveform(
        freq_contour=np.full(samples, 440),  # 440Hz
        t=np.linspace(0, 0.5, samples),
        samples=samples
    )
    
    # Apply simple envelope
    envelope = np.ones(samples)
    fade_length = samples // 20  # 5% fade
    envelope[:fade_length] *= np.linspace(0, 1, fade_length)
    envelope[-fade_length:] *= np.linspace(1, 0, fade_length)
    vowel_waveform *= envelope
    
    # Normalize
    if np.max(np.abs(vowel_waveform)) > 0:
        vowel_waveform = vowel_waveform / np.max(np.abs(vowel_waveform)) * 0.5
    
    sf.write("test_raw_vowel.wav", vowel_waveform, 22050)
    print(f"Raw vowel - Length: {len(vowel_waveform)}, Max: {np.max(np.abs(vowel_waveform)):.4f}")
    print(f"Silent samples: {np.sum(np.abs(vowel_waveform) < 0.001)} ({np.sum(np.abs(vowel_waveform) < 0.001)/len(vowel_waveform)*100:.1f}%)")
    
    # Test 2: Test diphone creation
    print("\n2. Testing diphone creation...")
    diphone = synth._create_synthetic_diphone("AH", "AH", "perfect_paul")
    sf.write("test_diphone_ah.wav", diphone.waveform, 22050)
    print(f"Diphone AH_AH - Length: {len(diphone.waveform)}, Max: {np.max(np.abs(diphone.waveform)):.4f}")
    print(f"Silent samples: {np.sum(np.abs(diphone.waveform) < 0.001)} ({np.sum(np.abs(diphone.waveform) < 0.001)/len(diphone.waveform)*100:.1f}%)")
    
    # Test 3: Test simple concatenation
    print("\n3. Testing simple concatenation...")
    phonemes = ["AH", "EH"]
    diphones = synth._phonemes_to_diphones(phonemes)
    print(f"Diphones: {diphones}")
    
    units = []
    for diphone_name in diphones:
        unit = synth._select_diphone_unit(diphone_name, synth._default_prosody())
        units.append(unit)
    
    # Simple concatenation without windowing
    simple_concat = np.concatenate([unit.waveform for unit in units])
    sf.write("test_simple_concat.wav", simple_concat, 22050)
    print(f"Simple concat - Length: {len(simple_concat)}, Max: {np.max(np.abs(simple_concat)):.4f}")
    print(f"Silent samples: {np.sum(np.abs(simple_concat) < 0.001)} ({np.sum(np.abs(simple_concat) < 0.001)/len(simple_concat)*100:.1f}%)")
    
    # Test 4: Test with minimal processing
    print("\n4. Testing with minimal processing...")
    minimal_result = synth._concatenate_diphones(units, synth._default_prosody())
    sf.write("test_minimal_processing.wav", minimal_result, 22050)
    print(f"Minimal processing - Length: {len(minimal_result)}, Max: {np.max(np.abs(minimal_result)):.4f}")
    print(f"Silent samples: {np.sum(np.abs(minimal_result) < 0.001)} ({np.sum(np.abs(minimal_result) < 0.001)/len(minimal_result)*100:.1f}%)")

if __name__ == "__main__":
    test_raw_synthesis()
