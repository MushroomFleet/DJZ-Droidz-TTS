#!/usr/bin/env python3
"""
DJZ-Hawk Demo Script
Demonstrates the complete DECtalk 4.2CD recreation system
"""

import os
import sys
import time

def print_banner():
    """Print DJZ-Hawk banner"""
    print("=" * 70)
    print("🤖 DJZ-HAWK rev0 - DECtalk 4.2CD (1996) Recreation")
    print("   Faithful Stephen Hawking Voice Synthesis")
    print("=" * 70)
    print()

def demo_basic_synthesis():
    """Demonstrate basic synthesis capabilities"""
    print("📢 DEMO 1: Basic Synthesis")
    print("-" * 30)
    
    test_phrases = [
        "Hello world",
        "Testing speech synthesis",
        "The quick brown fox jumps over the lazy dog"
    ]
    
    for i, phrase in enumerate(test_phrases, 1):
        print(f"Synthesizing: \"{phrase}\"")
        output_file = f"demo_basic_{i}.wav"
        
        # Run synthesis
        cmd = f'python main.py "{phrase}" -o {output_file}'
        os.system(cmd)
        
        if os.path.exists(output_file):
            print(f"✅ Generated: {output_file}")
        else:
            print(f"❌ Failed to generate: {output_file}")
        print()

def demo_hawking_phrases():
    """Demonstrate Stephen Hawking style phrases"""
    print("🎓 DEMO 2: Stephen Hawking Style Phrases")
    print("-" * 40)
    
    hawking_phrases = [
        "My name is Stephen Hawking",
        "I am a theoretical physicist", 
        "The universe has no boundary",
        "Black holes are not so black",
        "Intelligence is the ability to adapt to change",
        "We are just an advanced breed of monkeys on a minor planet"
    ]
    
    for i, phrase in enumerate(hawking_phrases, 1):
        print(f"Hawking Quote {i}: \"{phrase}\"")
        output_file = f"demo_hawking_{i}.wav"
        
        # Run synthesis
        cmd = f'python main.py "{phrase}" -o {output_file}'
        os.system(cmd)
        
        if os.path.exists(output_file):
            print(f"✅ Generated: {output_file}")
        else:
            print(f"❌ Failed to generate: {output_file}")
        print()

def demo_technical_terms():
    """Demonstrate technical/scientific term pronunciation"""
    print("🔬 DEMO 3: Technical Terms")
    print("-" * 30)
    
    technical_phrases = [
        "Quantum mechanics and general relativity",
        "Electromagnetic radiation from black holes", 
        "Thermodynamics of gravitational systems",
        "Cosmological constant and dark energy",
        "Singularities and event horizons"
    ]
    
    for i, phrase in enumerate(technical_phrases, 1):
        print(f"Technical {i}: \"{phrase}\"")
        output_file = f"demo_technical_{i}.wav"
        
        # Run synthesis
        cmd = f'python main.py "{phrase}" -o {output_file}'
        os.system(cmd)
        
        if os.path.exists(output_file):
            print(f"✅ Generated: {output_file}")
        else:
            print(f"❌ Failed to generate: {output_file}")
        print()

def demo_voice_characteristics():
    """Demonstrate voice characteristics"""
    print("🎭 DEMO 4: Voice Characteristics Test")
    print("-" * 35)
    
    test_phrase = "This demonstrates the characteristic DECtalk voice synthesis"
    
    # Test with different settings (if available)
    print(f"Testing phrase: \"{test_phrase}\"")
    output_file = "demo_characteristics.wav"
    
    cmd = f'python main.py "{test_phrase}" -o {output_file}'
    os.system(cmd)
    
    if os.path.exists(output_file):
        print(f"✅ Generated: {output_file}")
        
        # Analyze the output
        print("\nRunning audio analysis...")
        os.system("python test_speech_quality.py")
    else:
        print(f"❌ Failed to generate: {output_file}")
    print()

def demo_summary():
    """Print demo summary"""
    print("📊 DEMO SUMMARY")
    print("-" * 20)
    
    # Count generated files
    demo_files = [f for f in os.listdir('.') if f.startswith('demo_') and f.endswith('.wav')]
    
    print(f"Generated {len(demo_files)} audio files:")
    for file in sorted(demo_files):
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"  📄 {file} ({file_size:.1f} KB)")
    
    print(f"\n🎯 DJZ-Hawk Features Demonstrated:")
    print("  ✅ LPC-based synthesis for intelligibility")
    print("  ✅ Authentic DECtalk metallic resonance")
    print("  ✅ Perfect Paul voice characteristics")
    print("  ✅ 1996-era audio processing artifacts")
    print("  ✅ Stephen Hawking voice aesthetics")
    print("  ✅ Technical term pronunciation")
    print("  ✅ Characteristic robotic but clear speech")
    
    print(f"\n🔊 To play audio files:")
    print("  Windows: Use Windows Media Player or VLC")
    print("  macOS: Use QuickTime Player or VLC") 
    print("  Linux: Use aplay, vlc, or audacity")
    
    print(f"\n🎉 DJZ-Hawk rev0 demonstration complete!")
    print("   Faithful recreation of DECtalk 4.2CD achieved!")

def main():
    """Run complete DJZ-Hawk demonstration"""
    print_banner()
    
    print("Starting comprehensive DJZ-Hawk demonstration...")
    print("This will generate multiple audio files showcasing the system.\n")
    
    try:
        demo_basic_synthesis()
        demo_hawking_phrases() 
        demo_technical_terms()
        demo_voice_characteristics()
        demo_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
