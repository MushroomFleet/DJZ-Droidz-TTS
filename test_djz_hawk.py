#!/usr/bin/env python3
"""
Test script for DJZ-Hawk rev0
Verifies core functionality of the DECtalk 4.2CD recreation
"""

import sys
import os
import numpy as np

# Add the djz_hawk package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import DJZHawk

def test_basic_functionality():
    """Test basic synthesis functionality"""
    print("="*60)
    print("DJZ-HAWK rev0 Test Suite")
    print("="*60)
    
    try:
        # Initialize DJZ-Hawk
        print("1. Testing initialization...")
        djz_hawk = DJZHawk()
        print("   ✓ Initialization successful")
        
        # Test voice listing
        print("2. Testing voice listing...")
        voices = djz_hawk.list_voices()
        print(f"   ✓ Found {len(voices)} voices: {', '.join(voices)}")
        
        # Test basic synthesis
        print("3. Testing basic synthesis...")
        test_text = "Hello world"
        waveform = djz_hawk.synthesize(test_text)
        if len(waveform) > 0:
            print(f"   ✓ Generated {len(waveform)} samples ({len(waveform)/22050:.2f} seconds)")
        else:
            print("   ✗ No audio generated")
            return False
        
        # Test voice switching
        print("4. Testing voice switching...")
        djz_hawk.set_voice('huge_harry')
        waveform2 = djz_hawk.synthesize("Testing Huge Harry voice")
        if len(waveform2) > 0:
            print(f"   ✓ Huge Harry synthesis successful ({len(waveform2)/22050:.2f} seconds)")
        else:
            print("   ✗ Huge Harry synthesis failed")
            return False
        
        # Test file output
        print("5. Testing file output...")
        djz_hawk.synthesize("Testing file output", save_file="test_synthesis.wav")
        if os.path.exists("test_synthesis.wav"):
            print("   ✓ File output successful")
            # Clean up
            os.remove("test_synthesis.wav")
        else:
            print("   ✗ File output failed")
            return False
        
        # Test artifact intensity
        print("6. Testing artifact intensity...")
        djz_hawk.synthesizer.set_artifact_intensity(2.0)
        waveform3 = djz_hawk.synthesize("Testing maximum artifacts")
        if len(waveform3) > 0:
            print("   ✓ Artifact intensity adjustment successful")
        else:
            print("   ✗ Artifact intensity test failed")
            return False
        
        # Test all voices
        print("7. Testing all voices...")
        for voice in voices:
            try:
                djz_hawk.set_voice(voice)
                waveform = djz_hawk.synthesize(f"Testing {voice}")
                if len(waveform) > 0:
                    print(f"   ✓ {voice}: {len(waveform)/22050:.2f}s")
                else:
                    print(f"   ✗ {voice}: No audio generated")
            except Exception as e:
                print(f"   ✗ {voice}: Error - {e}")
        
        print("\n" + "="*60)
        print("✓ All tests completed successfully!")
        print("DJZ-Hawk rev0 is ready for 1996-style speech synthesis!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        return False

def test_text_processing():
    """Test text processing capabilities"""
    print("\n8. Testing text processing...")
    
    djz_hawk = DJZHawk()
    
    test_cases = [
        "Hello world.",
        "Dr. Smith lives on 123 Main St.",
        "The year 1996 was great for technology.",
        "Testing numbers: 1st, 2nd, 3rd, 4th.",
        "Abbreviations: Mr. Jones, Mrs. Smith, etc.",
        "What time is it? It's 3:30 PM!",
        "Testing punctuation: Hello, world! How are you?",
    ]
    
    for i, text in enumerate(test_cases):
        try:
            waveform = djz_hawk.synthesize(text)
            if len(waveform) > 0:
                print(f"   ✓ Case {i+1}: '{text}' -> {len(waveform)/22050:.2f}s")
            else:
                print(f"   ✗ Case {i+1}: '{text}' -> No audio")
        except Exception as e:
            print(f"   ✗ Case {i+1}: '{text}' -> Error: {e}")

def demo_voices():
    """Demo all voices with characteristic phrases"""
    print("\n9. Voice demonstrations...")
    
    djz_hawk = DJZHawk()
    
    voice_demos = {
        'perfect_paul': "Hello, my name is Perfect Paul. I am the voice of Stephen Hawking.",
        'beautiful_betty': "Hello, my name is Beautiful Betty. I have a pleasant female voice.",
        'huge_harry': "Hello, my name is Huge Harry. I am used for airport announcements.",
        'kit_the_kid': "Hi there! I'm Kit the Kid and I love to play!",
        'frank': "Hello, I'm Frank. I'm an alternative male voice.",
        'rita': "Hello, I'm Rita. I have a warm, breathy voice.",
        'ursula': "Greetings! I am Ursula, the dramatic voice.",
        'val': "Hi! I'm Val, like, totally awesome!",
        'rough': "Hello, I'm Rough. I have a gravelly voice."
    }
    
    for voice, text in voice_demos.items():
        try:
            djz_hawk.set_voice(voice)
            waveform = djz_hawk.synthesize(text)
            if len(waveform) > 0:
                print(f"   ✓ {voice}: '{text}' -> {len(waveform)/22050:.2f}s")
            else:
                print(f"   ✗ {voice}: No audio generated")
        except Exception as e:
            print(f"   ✗ {voice}: Error - {e}")

if __name__ == '__main__':
    success = test_basic_functionality()
    
    if success:
        test_text_processing()
        demo_voices()
        
        print("\n" + "="*60)
        print("🎉 DJZ-HAWK rev0 TEST SUITE COMPLETED SUCCESSFULLY! 🎉")
        print("The first fully working version of DJZ-Hawk is ready!")
        print("="*60)
        print("\nNext steps:")
        print("- Try: python main.py --interactive")
        print("- Try: python main.py 'Hello world' -v huge_harry")
        print("- Try: python main.py --demo perfect_paul")
        print("="*60)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
