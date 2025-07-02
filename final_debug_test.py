#!/usr/bin/env python3
"""
Final debug test to isolate the static pulse issue
"""

import numpy as np
import soundfile as sf
from djz_hawk.src.text_processor import DECtalk96TextProcessor
from djz_hawk.src.diphone_synthesizer import DECtalkDiphoneSynthesizer
from djz_hawk.src.vintage_artifacts import VintageArtifactGenerator

def analyze_audio(waveform, name):
    """Analyze audio waveform"""
    if len(waveform) == 0:
        print(f"{name}: EMPTY WAVEFORM")
        return
    
    length = len(waveform)
    duration = length / 22050
    max_amp = np.max(np.abs(waveform))
    rms = np.sqrt(np.mean(waveform**2))
    silent_samples = np.sum(np.abs(waveform) < 0.001)
    silent_percent = (silent_samples / length) * 100
    
    print(f"{name}:")
    print(f"  Length: {length} samples ({duration:.2f}s)")
    print(f"  Max amplitude: {max_amp:.4f}")
    print(f"  RMS: {rms:.4f}")
    print(f"  Silent samples: {silent_samples} ({silent_percent:.1f}%)")
    print()

def test_synthesis_pipeline():
    """Test each step of the synthesis pipeline"""
    print("Testing synthesis pipeline step by step...")
    
    text = "Hello"
    
    # Step 1: Text processing
    print("Step 1: Text processing")
    processor = DECtalk96TextProcessor()
    processed_segments = processor.process_text(text)
    print(f"Processed segments: {processed_segments}")
    
    # Step 2: Phoneme conversion
    print("Step 2: Phoneme conversion")
    all_phonemes = []
    for segment_text, context in processed_segments:
        phonemes = processor.text_to_phonemes(segment_text)
        all_phonemes.extend(phonemes)
    print(f"Phonemes: {all_phonemes}")
    
    # Step 3: Diphone synthesis
    print("Step 3: Diphone synthesis")
    synthesizer = DECtalkDiphoneSynthesizer()
    waveform_raw = synthesizer.synthesize_phoneme_sequence(all_phonemes)
    analyze_audio(waveform_raw, "After diphone synthesis")
    sf.write("debug_step3_diphone.wav", waveform_raw, 22050)
    
    # Step 4: Vintage artifacts
    print("Step 4: Vintage artifacts")
    artifact_gen = VintageArtifactGenerator()
    
    # Test each artifact step individually
    waveform_isa = artifact_gen.apply_isa_card_characteristics(waveform_raw.copy())
    analyze_audio(waveform_isa, "After ISA card characteristics")
    sf.write("debug_step4a_isa.wav", waveform_isa, 22050)
    
    # Test beeps (this might be the culprit)
    prosody_markers = [len(waveform_raw) - 1]  # End of speech
    waveform_beeps = artifact_gen.add_characteristic_beeps(waveform_isa.copy(), prosody_markers)
    analyze_audio(waveform_beeps, "After characteristic beeps")
    sf.write("debug_step4b_beeps.wav", waveform_beeps, 22050)
    
    # Test EQ
    waveform_eq = artifact_gen.apply_dectalk_eq_characteristics(waveform_beeps.copy())
    analyze_audio(waveform_eq, "After EQ characteristics")
    sf.write("debug_step4c_eq.wav", waveform_eq, 22050)
    
    print("Pipeline test complete. Check debug_step*.wav files.")

if __name__ == "__main__":
    test_synthesis_pipeline()
