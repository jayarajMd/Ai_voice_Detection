"""
Test AI voices with Deep Learning model active
"""

import logging
from pathlib import Path
from ai_voice_detector import AIVoiceDetector, AudioConfig

logging.basicConfig(level=logging.ERROR)

def test_ai_voices_with_ml():
    """Test AI voices with ML model"""
    
    ai_voices_dir = Path(r"c:\Users\mugam\Downloads\ai_voices")
    audio_files = sorted(list(ai_voices_dir.glob("*.mp3")))[:5]  # Test first 5
    
    print("\n" + "="*80)
    print("AI VOICE DETECTION - WITH DEEP LEARNING MODEL")
    print("="*80)
    print(f"Testing {len(audio_files)} samples with ML model active...\n")
    
    config = AudioConfig(target_sr=16000, ai_threshold=0.6, max_duration_sec=10.0)
    detector = AIVoiceDetector(config)
    
    results = []
    correct = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {audio_file.name}...", end=" ")
        
        try:
            result = detector.detect(str(audio_file))
            
            parts = audio_file.stem.split('_')
            language = parts[1] if len(parts) > 1 else "unknown"
            
            is_correct = result['label'] == 'AI_GENERATED'
            if is_correct:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            scores = result['individual_scores']
            
            results.append({
                'file': audio_file.name,
                'language': language,
                'label': result['label'],
                'score': result['final_score'],
                'ml_score': scores.get('ml', 'N/A'),
                'correct': is_correct
            })
            
            print(f"{status} {result['label']} (score: {result['final_score']:.3f}, ML: {scores.get('ml', 'N/A'):.3f})")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
    
    print("\nDetailed Breakdown:")
    print("-" * 80)
    print(f"{'File':<25} {'Language':<12} {'ML Score':<10} {'Final':<10} {'Result':<10}")
    print("-" * 80)
    
    for r in results:
        ml_score = f"{r['ml_score']:.3f}" if isinstance(r['ml_score'], float) else "N/A"
        status = "✅ CORRECT" if r['correct'] else "❌ WRONG"
        print(f"{r['file']:<25} {r['language'].capitalize():<12} {ml_score:<10} {r['score']:<10.3f} {status}")
    
    avg_ml = sum(r['ml_score'] for r in results if isinstance(r['ml_score'], float)) / len(results)
    avg_final = sum(r['score'] for r in results) / len(results)
    
    print("\n" + "="*80)
    print(f"Average ML Score:    {avg_ml:.3f}")
    print(f"Average Final Score: {avg_final:.3f}")
    print(f"Threshold:           0.600")
    print("="*80 + "\n")
    
    return results

if __name__ == "__main__":
    test_ai_voices_with_ml()
