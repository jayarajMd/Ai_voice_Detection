#!/usr/bin/env python3
"""
Test AI Voice Detection on ai_voices folder with Fingerprinting Technology
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ai_voice_detector import AIVoiceDetector

def test_ai_voices_folder():
    """Test all audio files in ai_voices folder."""
    
    ai_voices_dir = Path("ai_voices")
    
    if not ai_voices_dir.exists():
        print("[X] ERROR: ai_voices folder not found!")
        print("\nPlease place your AI-generated voice samples in the 'ai_voices' folder.")
        print("Supported formats: .wav, .mp3, .flac, .ogg")
        return
    
    # Find all audio files
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        audio_files.extend(list(ai_voices_dir.glob(f"*{ext}")))
    
    if not audio_files:
        print("[X] ERROR: No audio files found in ai_voices folder!")
        print("\nSupported formats: .wav, .mp3, .flac, .ogg")
        print("\nPlease add AI voice samples to the ai_voices folder and try again.")
        return
    
    print("="*80)
    print("AI VOICE DETECTION WITH ADVANCED FINGERPRINTING TECHNOLOGY")
    print("="*80)
    print(f"\nFound {len(audio_files)} audio file(s) in ai_voices folder")
    print("\nDetection System (9 Methods):")
    print("\n  Traditional Tests (60% total):")
    print("    [1] Pitch Stability       (8%)  - Analyzes voice naturalness")
    print("    [2] Spectral Smoothness   (10%) - Detects frequency patterns")
    print("    [3] Phase Artifacts       (10%) - Finds phase inconsistencies")
    print("    [4] Noise Randomness      (12%) - Measures noise entropy")
    print("    [5] Deep Learning         (20%) - Wav2Vec2 embeddings")
    print("\n  Fingerprinting Tests (40% total) - AI SIGNATURE DETECTION:")
    print("    [6] Spectral Fingerprint  (15%) - AI-specific frequency signatures")
    print("    [7] Temporal Fingerprint  (10%) - Unnatural timing consistency")
    print("    [8] Micro-Artifacts       (8%)  - Neural synthesis glitches")
    print("    [9] Prosody Fingerprint   (7%)  - Controlled rhythm/intonation")
    print("\nThreshold: 0.60 (scores >= 0.60 = AI-generated)")
    print("="*80)
    
    # Initialize detector
    print("\nInitializing detector...")
    detector = AIVoiceDetector()
    
    results = []
    correct_detections = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST {i}/{len(audio_files)}: {audio_file.name}")
        print('='*80)
        
        try:
            result = detector.detect(str(audio_file))
            
            score = result['final_score']
            is_ai = (result['label'] == 'AI_GENERATED')
            confidence = result['confidence']
            individual = result['individual_scores']
            
            # For AI voice samples, correct = detected as AI
            is_correct = is_ai
            if is_correct:
                correct_detections += 1
            
            print(f"\n{'='*80}")
            print(f"RESULT: {'[OK] CORRECTLY DETECTED AS AI' if is_correct else '[X] WRONGLY DETECTED AS HUMAN'}")
            print(f"{'='*80}")
            print(f"Final Score:     {score:.4f}")
            print(f"Classification:  {'AI-GENERATED' if is_ai else 'HUMAN'}")
            print(f"Confidence:      {confidence:.1%}")
            print(f"\nBreakdown of Individual Scores (0=Human, 1=AI):")
            print("  Traditional:")
            print(f"    Pitch:         {individual['pitch']:.3f}")
            print(f"    Spectral:      {individual['spectral']:.3f}")
            print(f"    Phase:         {individual['phase']:.3f}")
            print(f"    Noise:         {individual['noise']:.3f}")
            print(f"    Deep Learning: {individual['ml']:.3f}")
            print("  Fingerprinting:")
            print(f"    Spectral FP:   {individual['spectral_fp']:.3f}")
            print(f"    Temporal FP:   {individual['temporal_fp']:.3f}")
            print(f"    Micro-Artifact:{individual['micro_artifacts']:.3f}")
            print(f"    Prosody FP:    {individual['prosody_fp']:.3f}")
            print('='*80)
            
            results.append({
                'file': audio_file.name,
                'score': score,
                'detected_as_ai': is_ai,
                'confidence': confidence,
                'correct': is_correct,
                'individual': individual
            })
            
        except Exception as e:
            print(f"\n[X] ERROR processing {audio_file.name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'file': audio_file.name,
                'score': None,
                'detected_as_ai': False,
                'confidence': 0,
                'correct': False,
                'individual': None
            })
    
    # Final Summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    accuracy = (correct_detections / len(audio_files)) * 100
    
    print(f"\nTotal AI voice samples tested: {len(audio_files)}")
    print(f"Correctly detected as AI:      {correct_detections}")
    print(f"Incorrectly detected as HUMAN: {len(audio_files) - correct_detections}")
    print(f"\n[*] ACCURACY: {accuracy:.1f}%")
    
    print("\n" + "-"*80)
    print(f"{'File':<35} {'Score':<10} {'Detection':<15} {'Result'}")
    print("-"*80)
    for r in results:
        if r['score'] is not None:
            det = "AI" if r['detected_as_ai'] else "HUMAN"
            status = "[OK]" if r['correct'] else "[X]"
            print(f"{r['file']:<35} {r['score']:<10.4f} {det:<15} {status}")
        else:
            print(f"{r['file']:<35} {'ERROR':<10} {'N/A':<15} [X]")
    
    print("\n" + "="*80)
    print("ACCURACY ASSESSMENT")
    print("="*80)
    
    if accuracy >= 80:
        print("\n[OK] EXCELLENT: Fingerprinting technology is highly effective!")
        print("     The system successfully detects AI-generated voices.")
    elif accuracy >= 60:
        print("\n[!] GOOD: Fingerprinting shows significant improvement.")
        print("     Most AI voices are detected correctly.")
    elif accuracy >= 40:
        print("\n[!] MODERATE: System shows partial effectiveness.")
        print("     Further tuning may improve accuracy.")
    else:
        print("\n[X] POOR: Low detection rate.")
        print("     AI voices are difficult to distinguish with current weights.")
    
    # Analysis of best performing tests
    if results and results[0]['individual']:
        print("\n" + "="*80)
        print("TEST PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Average scores across all tests
        avg_scores = {}
        for test_name in results[0]['individual'].keys():
            scores = [r['individual'][test_name] for r in results if r['individual']]
            avg_scores[test_name] = sum(scores) / len(scores) if scores else 0
        
        print("\nAverage scores across all samples (higher = more AI-like):")
        print("  Traditional Tests:")
        for test in ['pitch', 'spectral', 'phase', 'noise', 'ml']:
            print(f"    {test:15s}: {avg_scores.get(test, 0):.3f}")
        print("  Fingerprinting Tests:")
        for test in ['spectral_fp', 'temporal_fp', 'micro_artifacts', 'prosody_fp']:
            print(f"    {test:15s}: {avg_scores.get(test, 0):.3f}")
        
        # Identify best performers
        sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nBest performing tests (most AI-sensitive):")
        for i, (test, score) in enumerate(sorted_scores[:3], 1):
            print(f"  {i}. {test}: {score:.3f}")
    
    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)

if __name__ == "__main__":
    test_ai_voices_folder()
