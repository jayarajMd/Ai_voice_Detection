# AI Voice Detection System - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 2: Prepare Test Audio

Option A: Use your own audio file (WAV or MP3)
```
Place your audio file in the same directory as ai_voice_detector.py
```

Option B: Generate test samples
```powershell
python test_detector.py
```
This creates synthetic test samples in `test_audio/` directory.

### Step 3: Run Detection

Create a file `example_usage.py`:

```python
from ai_voice_detector import AIVoiceDetector

# Initialize detector
detector = AIVoiceDetector()

# Run detection
result = detector.detect("your_audio.wav")

# Print results
print(f"Result: {result['label']}")
print(f"Confidence: {result['confidence']:.0%}")
```

Run it:
```powershell
python example_usage.py
```

---

## 📝 Example Output

```
================================================================================
AI VOICE DETECTION SYSTEM - FULL PIPELINE
================================================================================
Input: audio.wav

2026-02-05 10:30:15 - INFO - Loading audio from: audio.wav
2026-02-05 10:30:15 - INFO - ✓ Loaded: audio.wav | SR: 44100Hz | Shape: (132300,)

============================================================
STAGE 2: PREPROCESSING
============================================================
2026-02-05 10:30:15 - INFO - 1. Mono conversion: (132300,)
2026-02-05 10:30:15 - INFO - 2. Resampled to 16000Hz
2026-02-05 10:30:16 - INFO - 3. Silence trimmed (threshold: 20dB)
2026-02-05 10:30:16 - INFO - 4. Duration OK: 2.85s
2026-02-05 10:30:16 - INFO - 5. Amplitude normalized
2026-02-05 10:30:16 - INFO - 6. Denoising applied

📊 Waveform Statistics:
   Duration: 2.85s
   Samples: 45600
   RMS Energy: 0.0891
   Peak Amplitude: 0.9987
   Dynamic Range: 20.98dB

============================================================
STAGE 3: FEATURE EXTRACTION
============================================================
2026-02-05 10:30:16 - INFO - 1. MFCC: shape (40, 90)
2026-02-05 10:30:17 - INFO - 2. Pitch: 90 frames
2026-02-05 10:30:17 - INFO - 3. Energy: 90 frames
2026-02-05 10:30:17 - INFO - 4. Spectral centroid: 90 frames
2026-02-05 10:30:17 - INFO - 5. Spectral flatness: 90 frames
2026-02-05 10:30:17 - INFO - 6. HNR: 18.45 dB
2026-02-05 10:30:17 - INFO - 7. Phase features: 2 components
2026-02-05 10:30:17 - INFO - 8. Noise profile: 3 features
2026-02-05 10:30:18 - INFO - 9. Deep embeddings: shape (90, 768)

✓ Feature extraction complete

============================================================
STAGE 4: DETECTION TESTS
============================================================

1. PITCH STABILITY TEST
   Jitter: 0.008234
   Shimmer: 0.045123
   Pitch variance: 1234.56
   → AI probability: 0.245

2. SPECTRAL SMOOTHNESS TEST
   Spectral entropy: 5.234
   Harmonic variance: 2345678.90
   Flatness: 0.0812
   → AI probability: 0.187

3. PHASE ARTIFACT TEST
   Phase coherence: 0.5678
   Group delay variance: 0.002345
   → AI probability: 0.312

4. NOISE RANDOMNESS TEST
   Residual autocorr: 0.1823
   Residual entropy: 5.678
   → AI probability: 0.223

5. DEEP LEARNING TEST
   Embedding variance: 0.045678
   Temporal consistency: 0.8234
   → AI probability: 0.398

✓ All tests complete

============================================================
STAGE 5 & 6: NORMALIZATION AND FUSION
============================================================

📊 Normalized Scores (0 = Human, 1 = AI):
   Pitch Stability:      0.245
   Spectral Smoothness:  0.187
   Phase Artifacts:      0.312
   Noise Randomness:     0.223
   Deep Learning:        0.398

⚖️  Fusion Weights:
   Pitch:     15%
   Spectral:  20%
   Phase:     20%
   Noise:     10%
   ML:        35%

🎯 Final Fused Score: 0.2893
   (0.0 = Certainly Human, 1.0 = Certainly AI)

============================================================
STAGE 7: FINAL DECISION
============================================================

🏷️  Classification:
   Final Score:  0.2893
   Threshold:    0.6
   Decision:     HUMAN
   Confidence:   71%
   Certainty:    High

================================================================================
✓ DETECTION COMPLETE
================================================================================
```

---

## 🎨 Advanced Usage

### Custom Configuration

```python
from ai_voice_detector import AIVoiceDetector, AudioConfig

# Custom configuration
config = AudioConfig(
    max_duration_sec=20.0,      # Process up to 20 seconds
    ai_threshold=0.5,            # More sensitive detection
    weight_ml=0.50,              # Trust ML model more
    weight_pitch=0.10,
    weight_spectral=0.15,
    weight_phase=0.15,
    weight_noise=0.10,
    trim_db=25                   # More aggressive silence trimming
)

detector = AIVoiceDetector(config)
result = detector.detect("audio.wav")
```

### Batch Processing

```python
from pathlib import Path
from ai_voice_detector import AIVoiceDetector

detector = AIVoiceDetector()

# Process all WAV files in directory
audio_dir = Path("audio_samples/")
results = {}

for audio_file in audio_dir.glob("*.wav"):
    print(f"Processing: {audio_file.name}")
    
    try:
        result = detector.detect(str(audio_file))
        results[audio_file.name] = {
            'label': result['label'],
            'confidence': result['confidence'],
            'score': result['final_score']
        }
    except Exception as e:
        print(f"  Error: {e}")
        results[audio_file.name] = {'error': str(e)}

# Print summary
print("\n" + "="*60)
print("BATCH PROCESSING SUMMARY")
print("="*60)

ai_count = sum(1 for r in results.values() if r.get('label') == 'AI_GENERATED')
human_count = sum(1 for r in results.values() if r.get('label') == 'HUMAN')
error_count = sum(1 for r in results.values() if 'error' in r)

print(f"AI-generated: {ai_count}")
print(f"Human: {human_count}")
print(f"Errors: {error_count}")
print(f"Total: {len(results)}")

# Detailed results
print("\nDetailed Results:")
for filename, result in results.items():
    if 'error' in result:
        print(f"  ❌ {filename}: ERROR")
    else:
        confidence_str = f"{result['confidence']:.0%}"
        print(f"  {result['label']:15s} ({confidence_str:5s}): {filename}")
```

### From URL

```python
from ai_voice_detector import AIVoiceDetector

detector = AIVoiceDetector()

# Detect from URL
url = "https://example.com/sample_audio.wav"

try:
    result = detector.detect(url)
    print(f"Result: {result['label']} ({result['confidence']:.0%})")
except Exception as e:
    print(f"Error: {e}")
```

### Access Detailed Results

```python
from ai_voice_detector import AIVoiceDetector

detector = AIVoiceDetector()
result = detector.detect("audio.wav")

# Overall result
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Certainty: {result['certainty']}")

# Individual test scores
print("\nIndividual Test Scores:")
for test_name, score in result['individual_scores'].items():
    if score is not None:
        bar = "█" * int(score * 20)  # Visual bar
        print(f"  {test_name:20s}: {score:.3f} {bar}")

# Preprocessing statistics
print("\nPreprocessing Stats:")
for key, value in result['preprocessing_stats'].items():
    print(f"  {key}: {value}")

# Detailed test results
print("\nDetailed Test Results:")
for test_name, details in result['test_details'].items():
    print(f"\n{test_name}:")
    for metric, value in details.items():
        if metric != 'ai_probability':
            print(f"  {metric}: {value}")
```

---

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'transformers'**
```powershell
pip install transformers
```

**2. RuntimeError: ffmpeg not found**
- Download FFmpeg: https://ffmpeg.org/download.html
- Add to system PATH
- Verify: `ffmpeg -version`

**3. CUDA out of memory**
```python
# Force CPU usage
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

**4. Model download fails**
```python
# Download manually
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
```

**5. File not found**
```python
# Use absolute path
import os
audio_path = os.path.abspath("audio.wav")
result = detector.detect(audio_path)
```

---

## 📊 Understanding Results

### Score Interpretation

| Final Score | Meaning | Likely Classification |
|-------------|---------|----------------------|
| 0.0 - 0.3   | Very human-like | HUMAN (high confidence) |
| 0.3 - 0.5   | Probably human | HUMAN (moderate confidence) |
| 0.5 - 0.6   | Borderline | HUMAN (low confidence) |
| 0.6 - 0.7   | Borderline | AI_GENERATED (low confidence) |
| 0.7 - 0.8   | Probably AI | AI_GENERATED (moderate confidence) |
| 0.8 - 1.0   | Very AI-like | AI_GENERATED (high confidence) |

### Confidence Levels

- **Very High (>80%)**: Trust this result
- **High (60-80%)**: Reliable result
- **Moderate (40-60%)**: Consider context
- **Low (<40%)**: Uncertain, manual review recommended

### Individual Test Scores

Each test contributes to the final score:
- **Pitch**: Natural voice variations
- **Spectral**: Frequency domain characteristics
- **Phase**: Temporal coherence
- **Noise**: Background randomness
- **ML**: Learned patterns

High scores (>0.6) in multiple tests = stronger AI indication

---

## 🎯 Tips for Best Results

### Audio Quality
- Use clear, single-speaker audio
- Avoid heavy background noise
- Minimum 2 seconds of speech
- WAV format preferred (lossless)

### What to Avoid
- Multiple speakers (confuses analysis)
- Music with vocals (not designed for this)
- Very poor quality recordings
- Non-speech audio (animal sounds, etc.)

### When to Use
- ✅ Voice authentication
- ✅ Content moderation
- ✅ Deepfake detection
- ✅ Voice assistant verification
- ✅ Podcast/interview validation

### When NOT to Use
- ❌ Real-time detection (use streaming version)
- ❌ Music classification
- ❌ Non-English only (works but less accurate)
- ❌ Legal evidence (requires forensic validation)

---

## 📚 Next Steps

1. **Test with your audio samples**
2. **Adjust configuration for your use case**
3. **Integrate into your application**
4. **Deploy as API** (see SYSTEM_DOCUMENTATION.md)
5. **Collect feedback for improvement**

---

## 🤝 Need Help?

1. Check SYSTEM_DOCUMENTATION.md for detailed explanations
2. Run test suite: `python test_detector.py`
3. Review code comments in ai_voice_detector.py
4. Test with known samples to calibrate

---

**Ready to start? Run your first detection:**

```powershell
python -c "from ai_voice_detector import AIVoiceDetector; d = AIVoiceDetector(); print(d.detect('test_audio/test_complex.wav')['label'])"
```

Good luck! 🎉
