# DJZ-Hawk rev0: DECtalk 4.2CD (1996) Speech Synthesis Recreation

A faithful recreation of Digital Equipment Corporation's DECtalk 4.2CD speech synthesis system from 1996, representing the pinnacle of 1990s text-to-speech technology. This system recreates the distinctive robotic yet intelligible speech characteristics that defined Stephen Hawking's voice and 1990s computing.

## 🎯 Project Overview

DJZ-Hawk is a Python implementation that recreates the exact sound and characteristics of the original DECtalk 4.2CD system, including:

- **9 Authentic Voices**: Perfect Paul, Beautiful Betty, Huge Harry, Kit the Kid, Frank, Rita, Ursula, Val, and Rough
- **1996 Audio Characteristics**: ISA card processing, metallic resonance, concatenation artifacts
- **Period-Accurate Text Processing**: 1996-era abbreviation expansion and pronunciation rules
- **Diphone Concatenation**: Faithful recreation of DECtalk's synthesis methodology
- **Vintage Artifacts**: Electronic beeps, clicks, and characteristic audio processing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MushroomFleet/DJZ-Hawk
cd DJZ-Hawk

# Install dependencies
pip install -r requirements.txt

# Test the installation
python main.py --list-voices
```

### Basic Usage

```bash
# Speak with default voice (Perfect Paul)
python main.py "Hello, this is DJZ-Hawk speech synthesis"

# Use a different voice
python main.py "Hello world" --voice huge_harry

# Save to file
python main.py "Hello world" --output speech.wav

# Interactive mode
python main.py --interactive

# Demo a specific voice
python main.py --demo beautiful_betty
```

## 🗣️ Available Voices

| Voice | Description | Characteristics |
|-------|-------------|-----------------|
| **perfect_paul** | Default male voice (Stephen Hawking's voice) | 122Hz base, metallic resonance |
| **beautiful_betty** | Female voice based on Klatt's wife | 210Hz base, slight breathiness |
| **huge_harry** | Very deep male voice (airport ATIS) | 85Hz base, authoritative |
| **kit_the_kid** | Child voice based on Klatt's daughter | 280Hz base, slight lisp |
| **frank** | Alternative male voice | 115Hz base, clear articulation |
| **rita** | Warm female voice | 195Hz base, natural prosody |
| **ursula** | Dramatic female voice | 180Hz base, expressive |
| **val** | Valley girl style voice | 220Hz base, uptalk patterns |
| **rough** | Gravelly textured voice | 105Hz base, vocal fry |

## 🏗️ Architecture

```
djz_hawk/
├── src/
│   ├── text_processor.py      # 1996-era text normalization
│   ├── phoneme_engine.py      # ARPABET + context rules
│   ├── diphone_synthesizer.py # Core synthesis engine
│   ├── voice_models.py        # All 9 voice implementations
│   ├── prosody_engine.py      # Stress and intonation
│   ├── vintage_artifacts.py   # 1996 audio characteristics
│   └── audio_output.py        # Cross-platform audio
├── voices/                    # Voice databases
├── config/                    # Configuration files
├── tests/                     # Unit tests
├── examples/                  # Example scripts
└── main.py                    # Command-line interface
```

## 🎛️ Features

### Authentic 1996 Characteristics
- ✅ Distinctive robotic timbre with metallic resonance
- ✅ Characteristic concatenation artifacts and clicks
- ✅ Electronic beeps at phrase boundaries
- ✅ Dental stop assimilation (alveolar → dental)
- ✅ Limited prosodic variation
- ✅ ISA card audio processing simulation

### Text Processing
- Period-appropriate abbreviation expansion
- 1996-era number pronunciation rules
- Context-aware phoneme modifications
- DECtalk-specific pronunciation patterns

### Voice Synthesis
- Diphone concatenation methodology
- Formant-based voice modeling
- Linear Predictive Coding (LPC) processing
- Voice-specific characteristics and effects

## 📖 Usage Examples

### Command Line Interface

```bash
# Basic synthesis
python main.py "The quick brown fox jumps over the lazy dog"

# Voice comparison
python main.py "Hello, my name is Perfect Paul" --voice perfect_paul
python main.py "Hello, my name is Beautiful Betty" --voice beautiful_betty
python main.py "Hello, my name is Huge Harry" --voice huge_harry

# Save multiple voices
python main.py "Testing voice output" --voice perfect_paul --output paul.wav
python main.py "Testing voice output" --voice huge_harry --output harry.wav

# Interactive session
python main.py --interactive
```

### Interactive Mode Commands

```
DJZ-Hawk (perfect_paul)> Hello world
DJZ-Hawk (perfect_paul)> :voice huge_harry
DJZ-Hawk (huge_harry)> :demo beautiful_betty
DJZ-Hawk (huge_harry)> :save test.wav
DJZ-Hawk (huge_harry)> This will be saved to test.wav
DJZ-Hawk (huge_harry)> :quit
```

### Python API

```python
from main import DJZHawk

# Initialize synthesizer
hawk = DJZHawk(voice='perfect_paul')

# Synthesize speech
audio = hawk.synthesize("Hello from DJZ-Hawk")

# Change voice and speak
hawk.set_voice('huge_harry')
hawk.speak("This is Huge Harry speaking")

# Save to file
hawk.synthesize("Save this speech", save_file="output.wav")
```

## 🔧 Configuration

### Voice Configuration

Edit `config/voice_configs.json` to customize voice parameters:

```json
{
  "perfect_paul": {
    "base_frequency": 122.0,
    "frequency_range": 40.0,
    "roughness": 0.15,
    "timbre_metallic": 0.40,
    "speech_rate": 160
  }
}
```

### Global Settings

```json
{
  "global_settings": {
    "sample_rate": 22050,
    "vintage_artifacts_enabled": true,
    "concatenation_artifacts": true,
    "phrase_beeps": true,
    "metallic_resonance": true
  }
}
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_synthesis.py -v
python -m pytest tests/test_voices.py -v

# Performance benchmarks
python tests/benchmark_quality.py
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 2GB RAM minimum, 4GB recommended
- Working audio output device
- 500MB storage for voice databases

### Dependencies
- numpy >= 1.21.0
- scipy >= 1.7.0
- sounddevice >= 0.4.0
- soundfile >= 0.10.0
- librosa >= 0.8.0
- pyaudio >= 0.2.11

## 🎨 Customization

### Adding Custom Voices

```python
class CustomVoice(DECtalkVoiceModel):
    def __init__(self):
        characteristics = VoiceCharacteristics(
            name="Custom Voice",
            base_frequency=150.0,
            # ... other parameters
        )
        super().__init__(characteristics)
```

### Adjusting Vintage Artifacts

```python
# Increase 1996 artifacts
VINTAGE_ARTIFACT_LEVEL = 1.5  # 0.0 = clean, 2.0 = maximum

# Adjust concatenation artifacts
CONCATENATION_ARTIFACT_PROBABILITY = 0.8  # 80% chance per boundary
```

## 🔍 Troubleshooting

### Common Issues

**Audio playback not working:**
```bash
pip install sounddevice soundfile
# or use --output to save files instead
```

**Import errors:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/djz_hawk/src"
```

**Voice quality issues:**
- Check sample rate (should be 22050 Hz)
- Verify vintage artifacts are enabled
- Ensure proper voice configuration

### Performance Optimization

```bash
# Enable optimizations
export DJZ_HAWK_OPTIMIZE=1
export DJZ_HAWK_CACHE_SIZE=1000
export DJZ_HAWK_THREADS=4
```

## 📚 Historical Context

### DECtalk 4.2CD (1996)
- **Original System:** DECtalk PC (DTC-07) with DECtalk version 4.2CD
- **Architecture:** ISA card with dedicated CPU and RAM
- **Price:** $1,195 (1992 launch price)
- **Notable User:** Stephen Hawking (CallText 5010 variant)
- **Legacy:** Foundation for modern concatenative synthesis

### Technical Foundation
- **Synthesis Method:** Diphone concatenation
- **Voice Models:** Linear Predictive Coding (LPC) based
- **Foundation:** Dennis Klatt's formant synthesis (MITalk/KlattTalk)
- **Audio Quality:** 22.05kHz, 16-bit with characteristic artifacts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python -m pytest

# Format code
black djz_hawk/

# Lint code
flake8 djz_hawk/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dennis Klatt** - Original formant synthesis research at MIT
- **Digital Equipment Corporation** - Original DECtalk development
- **Stephen Hawking** - Making DECtalk famous worldwide
- **Vintage Computing Community** - Preservation of 1990s technology

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/MushroomFleet/DJZ-Hawk/issues)
- **Discussions:** [GitHub Discussions](https://github.com/MushroomFleet/DJZ-Hawk/discussions)
- **Documentation:** [Wiki](https://github.com/MushroomFleet/DJZ-Hawk/wiki)

---

**DJZ-Hawk rev0** - Bringing the distinctive sound of 1990s speech synthesis to the modern era.
