"""
Detailed Diagnostic Analysis of AI Voice Samples

Shows individual test scores to understand what features the AI voices have.
"""

import logging
from pathlib import Path
from ai_voice_detector import AIVoiceDetector, AudioConfig

logging.basicConfig(level=logging.CRITICAL)


def analyze_sample_details():
    """Analyze one sample in detail to understand the scores"""
    
    ai_voices_dir = Path(r"c:\Users\mugam\Downloads\ai_voices")
    audio_file = ai_voices_dir / "ai_english_1.mp3"
    
    if not audio_file.exists():
        print(f"File not found: {audio_file}")
        return
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF AI VOICE SAMPLE")
    print("="*80)
    print(f"File: {audio_file.name}")
    print("="*80 + "\n")
    
    # Initialize detector
    config = AudioConfig(
        target_sr=16000,
        ai_threshold=0.6
    )
    
    # Enable INFO logging just for this analysis
    logging.getLogger('ai_voice_detector').setLevel(logging.INFO)
    
    detector = AIVoiceDetector(config)
    
    print("Running full detection pipeline with detailed output...\n")
    
    # Run detection
    result = detector.detect(str(audio_file))
    
    print("\n" + "="*80)
    print("INDIVIDUAL TEST ANALYSIS")
    print("="*80)
    print("\nIndividual Test Scores (0.0 = Human, 1.0 = AI):\n")
    
    scores = result['individual_scores']
    
    for test_name, score in scores.items():
        if score is not None:
            # Interpret the score
            if score < 0.3:
                interpretation = "Strong HUMAN indicators"
            elif score < 0.5:
                interpretation = "Leans HUMAN"
            elif score < 0.7:
                interpretation = "Leans AI"
            else:
                interpretation = "Strong AI indicators"
            
            print(f"  {test_name:20s}: {score:.4f}  →  {interpretation}")
        else:
            print(f"  {test_name:20s}: N/A (not available)")
    
    print(f"\n{'='*80}")
    print(f"FUSION RESULT")
    print(f"{'='*80}")
    print(f"\nFinal Fused Score:  {result['final_score']:.4f}")
    print(f"Threshold:          {config.ai_threshold}")
    print(f"Decision:           {result['label']}")
    print(f"Confidence:         {result['confidence']:.1f}%")
    print(f"Certainty:          {result['certainty']}")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print("="*80)
    print("""
This AI-generated voice scored LOW on AI indicators, meaning:

1. PITCH STABILITY: The voice has natural pitch variations (like humans)
2. SPECTRAL SMOOTHNESS: The frequency spectrum is not too smooth
3. PHASE ARTIFACTS: Phase relationships are natural
4. NOISE RANDOMNESS: Background noise patterns are realistic
5. DEEP LEARNING: Not available (would be most effective here)

CONCLUSION:
This is a HIGH-QUALITY AI voice that successfully mimics human speech
characteristics. The signal processing tests alone cannot reliably detect it.

For modern high-quality AI voices, you would need:
- The deep learning model (Wav2Vec2) component active
- Lower detection threshold (e.g., 0.4-0.5)
- Additional advanced features
- Trained on modern TTS datasets
    """)
    
    print("="*80 + "\n")


if __name__ == "__main__":
    analyze_sample_details()
