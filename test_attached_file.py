"""
Test the attached audio file with detailed analysis
"""

import logging
from pathlib import Path
from ai_voice_detector import AIVoiceDetector, AudioConfig

# Enable INFO logging to see the full pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def test_attached_file():
    """Test the attached audio file"""
    
    audio_file = Path(r"c:\Users\mugam\Downloads\ac96ed42-c14b-4d80-bf67-aa80020c4d3a\ac96ed42-c14b-4d80-bf67-aa80020c4d3a.mp3")
    
    if not audio_file.exists():
        print(f"❌ File not found: {audio_file}")
        return
    
    print("\n" + "="*80)
    print(f"TESTING: {audio_file.name}")
    print("="*80)
    
    # Initialize detector with standard threshold
    config = AudioConfig(
        target_sr=16000,
        ai_threshold=0.6
    )
    
    detector = AIVoiceDetector(config)
    
    # Run detection
    result = detector.detect(str(audio_file))
    
    # Display detailed results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\n🎯 Prediction:  {result['label']}")
    print(f"📊 Confidence:  {result['confidence']:.1f}%")
    print(f"🔢 Score:       {result['final_score']:.4f}")
    print(f"⭐ Certainty:   {result['certainty']}")
    
    print("\n" + "-"*80)
    print("Individual Test Scores (0.0 = Human, 1.0 = AI):")
    print("-"*80)
    
    scores = result['individual_scores']
    for test_name, score in scores.items():
        if score is not None:
            bar_length = int(score * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            # Interpretation
            if score < 0.3:
                interpret = "Strong HUMAN"
            elif score < 0.5:
                interpret = "Leans HUMAN"
            elif score < 0.7:
                interpret = "Leans AI"
            else:
                interpret = "Strong AI"
            
            print(f"  {test_name:20s}: {score:.4f} {bar} {interpret}")
        else:
            print(f"  {test_name:20s}: N/A")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if result['label'] == 'AI_GENERATED':
        print(f"""
✅ This audio is classified as AI-GENERATED

The detection score ({result['final_score']:.4f}) is above the threshold ({config.ai_threshold}),
indicating that the voice exhibits characteristics typical of AI-generated speech.

Key indicators detected:
""")
        for test_name, score in scores.items():
            if score is not None and score > 0.6:
                print(f"  • {test_name}: {score:.4f} (AI indicator)")
    else:
        print(f"""
ℹ️  This audio is classified as HUMAN

The detection score ({result['final_score']:.4f}) is below the threshold ({config.ai_threshold}),
indicating that the voice exhibits characteristics typical of human speech.

This could mean:
  • The voice is genuinely human
  • It's a very high-quality AI voice that mimics human characteristics well
  • The signal processing tests alone cannot detect this level of AI sophistication
""")
        # Show if any tests suggested AI
        ai_indicators = [(name, score) for name, score in scores.items() 
                         if score is not None and score > 0.6]
        if ai_indicators:
            print("However, some tests showed AI indicators:")
            for test_name, score in ai_indicators:
                print(f"  • {test_name}: {score:.4f}")
    
    print("\n" + "="*80 + "\n")
    
    return result


if __name__ == "__main__":
    test_attached_file()
