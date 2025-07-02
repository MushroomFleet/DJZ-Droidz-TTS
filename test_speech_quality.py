#!/usr/bin/env python3
"""
Test script to verify DJZ-Hawk speech quality and characteristics
"""

import os
import sys
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def analyze_audio_file(filename):
    """Analyze audio file for DECtalk characteristics"""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    
    # Read audio file
    sample_rate, audio_data = wavfile.read(filename)
    
    # Convert to float if needed
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    
    print(f"\n=== Audio Analysis: {filename} ===")
    print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Samples: {len(audio_data)}")
    print(f"Max Amplitude: {np.max(np.abs(audio_data)):.3f}")
    print(f"RMS Level: {np.sqrt(np.mean(audio_data**2)):.3f}")
    
    # Check for silence
    silence_threshold = 0.01
    non_silent_samples = np.sum(np.abs(audio_data) > silence_threshold)
    speech_ratio = non_silent_samples / len(audio_data)
    print(f"Speech Activity: {speech_ratio:.1%}")
    
    # Frequency analysis
    fft = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
    magnitude = np.abs(fft)
    
    # Find peak frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]
    
    # Find peaks in different frequency ranges
    low_range = (positive_freqs >= 100) & (positive_freqs <= 500)
    mid_range = (positive_freqs >= 1000) & (positive_freqs <= 3000)
    high_range = (positive_freqs >= 3000) & (positive_freqs <= 8000)
    
    if np.any(low_range):
        low_peak_idx = np.argmax(positive_magnitude[low_range])
        low_peak_freq = positive_freqs[low_range][low_peak_idx]
        print(f"Low Frequency Peak: {low_peak_freq:.0f} Hz")
    
    if np.any(mid_range):
        mid_peak_idx = np.argmax(positive_magnitude[mid_range])
        mid_peak_freq = positive_freqs[mid_range][mid_peak_idx]
        print(f"Mid Frequency Peak: {mid_peak_freq:.0f} Hz")
    
    if np.any(high_range):
        high_peak_idx = np.argmax(positive_magnitude[high_range])
        high_peak_freq = positive_freqs[high_range][high_peak_idx]
        print(f"High Frequency Peak: {high_peak_freq:.0f} Hz")
    
    # Check for DECtalk characteristics
    print("\n=== DECtalk Characteristics ===")
    
    # Check for metallic resonance around 3.2kHz
    metallic_range = (positive_freqs >= 3000) & (positive_freqs <= 3500)
    if np.any(metallic_range):
        metallic_energy = np.mean(positive_magnitude[metallic_range])
        total_energy = np.mean(positive_magnitude)
        metallic_ratio = metallic_energy / total_energy if total_energy > 0 else 0
        print(f"Metallic Resonance (3-3.5kHz): {metallic_ratio:.2f} (higher = more metallic)")
    
    # Check for fundamental frequency around 122Hz (Perfect Paul)
    f0_range = (positive_freqs >= 100) & (positive_freqs <= 150)
    if np.any(f0_range):
        f0_energy = np.mean(positive_magnitude[f0_range])
        print(f"F0 Energy (100-150Hz): {f0_energy:.3f}")
    
    # Check for formant structure
    formant_ranges = [
        (400, 800, "F1"),
        (1200, 2000, "F2"), 
        (2200, 3200, "F3")
    ]
    
    for low, high, name in formant_ranges:
        formant_range = (positive_freqs >= low) & (positive_freqs <= high)
        if np.any(formant_range):
            formant_energy = np.mean(positive_magnitude[formant_range])
            print(f"{name} Energy ({low}-{high}Hz): {formant_energy:.3f}")
    
    return audio_data, sample_rate

def test_all_files():
    """Test all generated audio files"""
    test_files = [
        "test_lpc_synthesis.wav",
        "test_lpc_longer.wav", 
        "test_hawking_style.wav"
    ]
    
    print("DJZ-Hawk Speech Quality Analysis")
    print("=" * 50)
    
    for filename in test_files:
        if os.path.exists(filename):
            analyze_audio_file(filename)
        else:
            print(f"\nFile not found: {filename}")
    
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("\nExpected DECtalk Characteristics:")
    print("- Metallic resonance ratio > 1.0")
    print("- Clear formant structure in F1, F2, F3 ranges")
    print("- F0 energy around 122Hz for Perfect Paul")
    print("- Speech activity > 60%")
    print("- Characteristic robotic but intelligible sound")

if __name__ == "__main__":
    test_all_files()
