#!/usr/bin/env python3
"""
DJZ-DroidTTS - Advanced Text-to-Droid Speech Synthesis
Command-line interface for authentic droid and robot speech generation
"""

import argparse
import sys
import os
from typing import List, Optional
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'djz_hawk', 'src'))

try:
    from text_processor import DECtalk96TextProcessor
    from phoneme_engine import DECtalkPhonemeEngine
    from diphone_synthesizer import DECtalkDiphoneSynthesizer
    from voice_models import DECtalkVoiceManager
    from vintage_artifacts import VintageArtifactGenerator
    from prosody_engine import DECtalkProsodyEngine
    from audio_output import AudioOutput
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all required modules are installed and the src directory exists.")
    sys.exit(1)

class DJZHawk:
    """Main DJZ-Hawk speech synthesis engine"""
    
    def __init__(self, voice: str = 'perfect_paul'):
        try:
            self.text_processor = DECtalk96TextProcessor()
            self.phoneme_engine = DECtalkPhonemeEngine()
            self.voice_manager = DECtalkVoiceManager()
            self.synthesizer = DECtalkDiphoneSynthesizer(voice)
            self.artifact_generator = VintageArtifactGenerator()
            self.prosody_engine = DECtalkProsodyEngine()
            self.audio_output = AudioOutput()
            self.current_voice = voice
        except Exception as e:
            print(f"Error initializing DJZ-Hawk: {e}")
            raise
        
    def synthesize(self, text: str, voice: Optional[str] = None, 
                  save_file: Optional[str] = None) -> np.ndarray:
        """
        Synthesize speech from text using DECtalk 4.2CD methodology
        
        Args:
            text: Input text to synthesize
            voice: Voice name (optional, uses current voice if None)
            save_file: Optional file path to save audio
            
        Returns:
            Generated audio as numpy array
        """
        if voice and voice != self.current_voice:
            self.set_voice(voice)
        
        print(f"[DJZ-Hawk] Processing text with voice '{self.current_voice}'...")
        
        try:
            # 1. Process text with 1996-era rules
            processed_segments = self.text_processor.process_text(text)
            
            # 2. Convert to phonemes
            all_phonemes = []
            prosody_markers = []
            
            for segment_text, context in processed_segments:
                phonemes = self.text_processor.text_to_phonemes(segment_text)
                all_phonemes.extend(phonemes)
                
                # Add prosody based on context
                if hasattr(context, 'punctuation') and context.punctuation in ['.', '!', '?']:
                    prosody_markers.append(len(all_phonemes) - 1)  # Sentence boundary
            
            # 3. Synthesize using diphone concatenation
            print(f"[DJZ-Hawk] Synthesizing {len(all_phonemes)} phonemes...")
            waveform = self.synthesizer.synthesize_phoneme_sequence(all_phonemes)
            
            # 6. Apply characteristic DECtalk artifacts
            print(f"[DJZ-Hawk] Applying 1996 audio characteristics...")
            waveform = self.artifact_generator.apply_isa_card_characteristics(waveform)
            waveform = self.artifact_generator.add_characteristic_beeps(waveform, prosody_markers)
            
            # 7. Final vintage processing
            waveform = self.artifact_generator.apply_dectalk_eq_characteristics(waveform)
            
            # 8. Save or return
            if save_file:
                self.audio_output.save_wav(waveform, save_file)
                print(f"[DJZ-Hawk] Audio saved to {save_file}")
            
            return waveform
            
        except Exception as e:
            print(f"Error during synthesis: {e}")
            raise
    
    def set_voice(self, voice_name: str):
        """Change the current voice"""
        try:
            self.voice_manager.set_voice(voice_name)
            self.synthesizer = DECtalkDiphoneSynthesizer(voice_name)
            self.current_voice = voice_name
            print(f"[DJZ-Hawk] Voice changed to '{voice_name}'")
        except Exception as e:
            print(f"Error setting voice: {e}")
            raise
    
    def list_voices(self) -> List[str]:
        """List all available voices"""
        return self.voice_manager.list_voices()
    
    def speak(self, text: str, voice: Optional[str] = None):
        """Synthesize and play speech"""
        try:
            waveform = self.synthesize(text, voice)
            print(f"[DJZ-Hawk] Playing synthesized speech...")
            self.audio_output.play(waveform)
        except Exception as e:
            print(f"Error during speech playback: {e}")
            raise
    
    def demo_voice(self, voice_name: str):
        """Play demonstration of specific voice"""
        try:
            voice_model = self.voice_manager.get_voice(voice_name)
            test_phrase = voice_model.characteristics.test_phrase
            print(f"[DJZ-Hawk] Demo: {voice_name}")
            print(f"[DJZ-Hawk] Text: \"{test_phrase}\"")
            self.speak(test_phrase, voice_name)
        except Exception as e:
            print(f"Error during voice demo: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(
        description='DJZ-DroidTTS: Advanced Text-to-Droid Speech Synthesis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Greetings, human. I am a protocol droid."           # Basic droid speech
  %(prog)s "Affirmative. Mission parameters received." -v huge_harry  # Military droid
  %(prog)s "Warning: System malfunction detected." -o droid_alert.wav # Save to file
  %(prog)s --demo perfect_paul                                  # Demo protocol droid
  %(prog)s --interactive                                        # Interactive droid mode
  %(prog)s --list-voices                                        # List all droid voices
  
Available droid voices:
  perfect_paul (Protocol Droid), beautiful_betty (Service Droid), huge_harry (Security Droid),
  kit_the_kid (Utility Droid), frank (Standard Droid), rita (Companion Droid),
  ursula (Command Droid), val (Entertainment Droid), rough (Industrial Droid)
        """
    )
    
    parser.add_argument('text', nargs='?', help='Text to synthesize')
    parser.add_argument('-v', '--voice', default='perfect_paul',
                       help='Voice to use (default: perfect_paul)')
    parser.add_argument('-o', '--output', help='Output WAV file')
    parser.add_argument('-r', '--rate', type=int, default=22050,
                       help='Sample rate (default: 22050)')
    parser.add_argument('--demo', metavar='VOICE',
                       help='Demo specific voice')
    parser.add_argument('--list-voices', action='store_true',
                       help='List available voices')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--play', action='store_true', default=True,
                       help='Play audio (default: true)')
    parser.add_argument('--no-play', dest='play', action='store_false',
                       help='Don\'t play audio')
    
    args = parser.parse_args()
    
    # Initialize DJZ-Hawk
    try:
        djz_hawk = DJZHawk(voice=args.voice)
    except Exception as e:
        print(f"Error initializing DJZ-Hawk: {e}")
        sys.exit(1)
    
    # Handle commands
    if args.list_voices:
        print("Available voices:")
        for voice in djz_hawk.list_voices():
            try:
                voice_info = djz_hawk.voice_manager.get_voice_info(voice)
                print(f"  {voice:15} - {voice_info.name} ({voice_info.gender}, {voice_info.age_group})")
            except:
                print(f"  {voice:15} - (info unavailable)")
        return
    
    if args.demo:
        if args.demo not in djz_hawk.list_voices():
            print(f"Error: Unknown voice '{args.demo}'")
            print("Use --list-voices to see available voices")
            sys.exit(1)
        djz_hawk.demo_voice(args.demo)
        return
    
    if args.interactive:
        interactive_mode(djz_hawk)
        return
    
    if not args.text:
        parser.print_help()
        sys.exit(1)
    
    # Synthesize text
    try:
        waveform = djz_hawk.synthesize(args.text, args.voice, args.output)
        
        if args.play and not args.output:
            djz_hawk.audio_output.play(waveform)
            
    except Exception as e:
        print(f"Error during synthesis: {e}")
        sys.exit(1)

def interactive_mode(djz_hawk: DJZHawk):
    """Interactive mode for DJZ-DroidTTS"""
    print("="*60)
    print("DJZ-DroidTTS Interactive Mode")
    print("Advanced Text-to-Droid Speech Synthesis")
    print("="*60)
    print("Commands:")
    print("  :voice <name>     - Change droid voice")
    print("  :voices           - List droid voices") 
    print("  :demo <voice>     - Demo droid voice")
    print("  :save <file>      - Save next synthesis to file")
    print("  :quit             - Exit")
    print("  <text>            - Generate droid speech")
    print("="*60)
    
    save_next = None
    
    while True:
        try:
            user_input = input(f"DJZ-DroidTTS ({djz_hawk.current_voice})> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in [':quit', ':exit', ':q']:
                print("Goodbye!")
                break
                
            elif user_input.startswith(':voice '):
                voice_name = user_input[7:].strip()
                if voice_name in djz_hawk.list_voices():
                    djz_hawk.set_voice(voice_name)
                else:
                    print(f"Unknown voice: {voice_name}")
                    
            elif user_input == ':voices':
                print("Available voices:")
                for voice in djz_hawk.list_voices():
                    marker = " *" if voice == djz_hawk.current_voice else ""
                    print(f"  {voice}{marker}")
                    
            elif user_input.startswith(':demo '):
                voice_name = user_input[6:].strip()
                if voice_name in djz_hawk.list_voices():
                    djz_hawk.demo_voice(voice_name)
                else:
                    print(f"Unknown voice: {voice_name}")
                    
            elif user_input.startswith(':save '):
                save_next = user_input[6:].strip()
                print(f"Next synthesis will be saved to: {save_next}")
                
            elif user_input.startswith(':'):
                print("Unknown command")
                
            else:
                # Synthesize speech
                djz_hawk.speak(user_input)
                if save_next:
                    djz_hawk.synthesize(user_input, save_file=save_next)
                    save_next = None
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
