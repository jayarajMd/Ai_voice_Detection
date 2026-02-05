"""
Quick Test of AI Voice Detection System on Real AI-Generated Samples

Tests a subset of AI voice samples with streamlined output.
"""

import logging
from pathlib import Path
from ai_voice_detector import AIVoiceDetector, AudioConfig
import json
from datetime import datetime

# Minimal logging to avoid clutter
logging.basicConfig(level=logging.ERROR)


def test_ai_voice_samples(num_samples=5):
    """Test AI voice samples with clean output"""
    
    # Path to the ai_voices folder
    ai_voices_dir = Path(r"c:\Users\mugam\Downloads\ai_voices")
    
    if not ai_voices_dir.exists():
        print(f"❌ Folder not found: {ai_voices_dir}")
        return
    
    # Get all MP3 files
    audio_files = sorted(list(ai_voices_dir.glob("*.mp3")))
    
    if not audio_files:
        print(f"❌ No audio files found in {ai_voices_dir}")
        return
    
    # Limit to num_samples for quick testing
    audio_files = audio_files[:num_samples]
    
    print("\n" + "="*80)
    print("AI VOICE DETECTION SYSTEM - REAL SAMPLE TEST")
    print("="*80)
    print(f"Testing {len(audio_files)} AI-generated voice samples...")
    print(f"Location: {ai_voices_dir}")
    print("="*80)
    
    # Initialize detector with faster settings
    config = AudioConfig(
        target_sr=16000,
        ai_threshold=0.6,
        max_duration_sec=10.0  # Limit duration for faster processing
    )
    
    # Suppress all logging from detector
    logging.getLogger('ai_voice_detector').setLevel(logging.CRITICAL)
    
    detector = AIVoiceDetector(config)
    
    # Store results
    results = []
    correct = 0
    
    # Test each file
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Testing: {audio_file.name}...")
        
        try:
            # Run detection
            result = detector.detect(str(audio_file))
            
            # Extract language from filename (e.g., ai_english_1.mp3 → english)
            parts = audio_file.stem.split('_')
            language = parts[1] if len(parts) > 1 else "unknown"
            sample_num = parts[2] if len(parts) > 2 else "?"
            
            # Check correctness (all should be AI_GENERATED)
            is_correct = result['label'] == 'AI_GENERATED'
            if is_correct:
                correct += 1
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
            
            # Store result
            results.append({
                'filename': audio_file.name,
                'language': language,
                'sample': sample_num,
                'label': result['label'],
                'confidence': result['confidence'],
                'score': result['final_score'],
                'correct': is_correct
            })
            
            # Display result
            print(f"   Language:    {language.upper()}")
            print(f"   Prediction:  {result['label']}")
            print(f"   Confidence:  {result['confidence']:.1f}%")
            print(f"   Score:       {result['final_score']:.4f}")
            print(f"   Result:      {status}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append({
                'filename': audio_file.name,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Tested:        {len(results)}")
    print(f"Correctly Detected:  {correct}")
    print(f"Misclassified:       {len(results) - correct}")
    print(f"Accuracy:            {correct/len(results)*100:.1f}%")
    
    # Group by language
    print("\n" + "-"*80)
    print("BY LANGUAGE:")
    print("-"*80)
    
    by_lang = {}
    for r in results:
        if 'error' not in r:
            lang = r['language']
            if lang not in by_lang:
                by_lang[lang] = {'total': 0, 'correct': 0, 'scores': []}
            by_lang[lang]['total'] += 1
            if r['correct']:
                by_lang[lang]['correct'] += 1
            by_lang[lang]['scores'].append(r['score'])
    
    for lang, stats in sorted(by_lang.items()):
        acc = stats['correct'] / stats['total'] * 100
        avg_score = sum(stats['scores']) / len(stats['scores'])
        print(f"{lang.capitalize():12} - {stats['correct']}/{stats['total']} correct ({acc:.0f}%) | Avg Score: {avg_score:.3f}")
    
    # Save report
    report = {
        'test_date': datetime.now().isoformat(),
        'total_tested': len(results),
        'correct': correct,
        'accuracy': correct / len(results) * 100,
        'results': results
    }
    
    report_file = "ai_voices_quick_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Report saved to: {report_file}")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    # Test 5 samples by default (change number as needed)
    test_ai_voice_samples(num_samples=15)  # Test all 15
