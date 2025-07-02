#!/usr/bin/env python3
"""
Test script for DJZ-Hawk voices
Tests all 9 voices with their characteristic phrases
"""

from main import DJZHawk
import os

def test_all_voices():
    """Test all voices with their characteristic phrases"""
    print("="*60)
    print("DJZ-HAWK VOICE TEST")
    print("Testing all 9 DECtalk voices")
    print("="*60)
    
    # Initialize DJZ-Hawk
    hawk = DJZHawk()
    
    # Get all available voices
    voices = hawk.list_voices()
    
    # Test each voice
    for voice_name in voices:
        print(f"\nTesting voice: {voice_name}")
        
        try:
            # Get voice characteristics
            voice_info = hawk.voice_manager.get_voice_info(voice_name)
            test_phrase = voice_info.test_phrase
            
            print(f"  Name: {voice_info.name}")
            print(f"  Gender: {voice_info.gender}")
            print(f"  Base Frequency: {voice_info.base_frequency}Hz")
            print(f"  Test Phrase: \"{test_phrase}\"")
            
            # Set voice and synthesize
            hawk.set_voice(voice_name)
            
            # Generate audio file
            output_file = f"test_output_{voice_name}.wav"
            waveform = hawk.synthesize(test_phrase, save_file=output_file)
            
            print(f"  ✓ Generated: {output_file}")
            print(f"  ✓ Duration: {len(waveform)/22050:.2f} seconds")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Voice test complete!")
    print("Check the generated .wav files to hear each voice.")
    print("="*60)

def test_text_processing():
    """Test text processing with various inputs"""
    print("\n" + "="*60)
    print("TEXT PROCESSING TEST")
    print("="*60)
    
    hawk = DJZHawk()
    
    test_texts = [
        "Hello world!",
        "The year 1996 was great.",
        "Dr. Smith lives on 123 Main St.",
        "I can't believe it's working!",
        "Testing numbers: 1st, 2nd, 3rd place.",
        "Abbreviations: CPU, RAM, SCSI drive.",
        "Symbols: @ # $ % & *",
        "Multiple sentences. This is sentence two! Is this working?"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: \"{text}\"")
        try:
            waveform = hawk.synthesize(text, save_file=f"text_test_{i}.wav")
            print(f"  ✓ Generated: text_test_{i}.wav")
            print(f"  ✓ Duration: {len(waveform)/22050:.2f} seconds")
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_stephen_hawking_tribute():
    """Test with Stephen Hawking-style phrases"""
    print("\n" + "="*60)
    print("STEPHEN HAWKING TRIBUTE TEST")
    print("="*60)
    
    hawk = DJZHawk(voice='perfect_paul')
    
    hawking_phrases = [
        "My goal is simple. It is a complete understanding of the universe.",
        "Intelligence is the ability to adapt to change.",
        "We are just an advanced breed of monkeys on a minor planet.",
        "The greatest enemy of knowledge is not ignorance, it is the illusion of knowledge.",
        "Life would be tragic if it weren't funny."
    ]
    
    for i, phrase in enumerate(hawking_phrases, 1):
        print(f"\nHawking Quote {i}:")
        print(f"  \"{phrase}\"")
        try:
            waveform = hawk.synthesize(phrase, save_file=f"hawking_{i}.wav")
            print(f"  ✓ Generated: hawking_{i}.wav")
            print(f"  ✓ Duration: {len(waveform)/22050:.2f} seconds")
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    # Run all tests
    test_all_voices()
    test_text_processing()
    test_stephen_hawking_tribute()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE!")
    print("DJZ-Hawk rev0 is ready for use!")
    print("="*60)
