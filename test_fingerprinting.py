#!/usr/bin/env python3
"""
Test AI Voice Detection with Advanced Fingerprinting
Tests the 9-method system on AI-generated voice samples.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ai_voice_detector import AIVoiceDetector

def test_ai_voices():
    """Test multiple AI voice samples with fingerprinting."""
    
    # Initialize detector with fingerprinting enabled
    detector = AIVoiceDetector()
    
    # Test files
    test_folder = Path("test_audio")
    if not test_folder.exists():
        print("❌ test_audio folder not found")
        return
    
    ai_files = list(test_folder.glob("*.wav"))
    if not ai_files:
        print("❌ No WAV files found in test_audio folder")
        return
    
    print("="*70)
    print("TESTING AI VOICE DETECTION WITH FINGERPRINTING TECHNOLOGY")
    print("="*70)
    print(f"\nFound {len(ai_files)} AI voice samples")
    print("\nDetection Methods:")
    print("  Traditional Tests (60%):")
    print("    - Pitch Stability (8%)")
    print("    - Spectral Smoothness (10%)")
    print("    - Phase Artifacts (10%)")
    print("    - Noise Randomness (12%)")
    print("    - Deep Learning (20%)")
    print("  Fingerprinting Tests (40%):")
    print("    - Spectral Fingerprint (15%) - AI frequency signatures")
    print("    - Temporal Fingerprint (10%) - Unnatural timing")
    print("    - Micro-Artifacts (8%) - Neural synthesis glitches")
    print("    - Prosody Fingerprint (7%) - Controlled rhythm/intonation")
    print("\nThreshold: 0.6 (scores >= 0.6 = AI-generated)")
    print("="*70)
    
    results = []
    correct = 0
    
    for i, audio_file in enumerate(ai_files, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST {i}/{len(ai_files)}: {audio_file.name}")
        print('='*70)
        
        try:
            result = detector.detect(str(audio_file))
            
            score = result['final_score']
            is_ai = result['is_ai_generated']
            confidence = result['confidence']
            
            # Since these are all AI voices, correct detection is is_ai == True
            if is_ai:
                correct += 1
                status = "[OK] CORRECT"
            else:
                status = "[X] WRONG"
            
            print(f"\n{'='*70}")
            print(f"RESULT: {status}")
            print(f"Score: {score:.4f} -> {'AI-GENERATED' if is_ai else 'HUMAN'} ({confidence:.1%} confident)")
            print('='*70)
            
            results.append({
                'file': audio_file.name,
                'score': score,
                'detected_as_ai': is_ai,
                'correct': is_ai
            })
            
        except Exception as e:
            print(f"[X] ERROR: {e}")
            results.append({
                'file': audio_file.name,
                'score': None,
                'detected_as_ai': False,
                'correct': False
            })
    
    # Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    accuracy = (correct / len(ai_files)) * 100
    
    print(f"\nTotal samples: {len(ai_files)}")
    print(f"Correctly detected as AI: {correct}")
    print(f"Incorrectly detected as HUMAN: {len(ai_files) - correct}")
    print(f"\n[*] ACCURACY: {accuracy:.1f}%")
    
    print("\nDetailed Results:")
    print(f"{'File':<30} {'Score':<10} {'Detection':<12} {'Status'}")
    print("-"*70)
    for r in results:
        if r['score'] is not None:
            detection = "AI" if r['detected_as_ai'] else "HUMAN"
            status = "[OK]" if r['correct'] else "[X]"
            print(f"{r['file']:<30} {r['score']:<10.4f} {detection:<12} {status}")
        else:
            print(f"{r['file']:<30} {'ERROR':<10} {'N/A':<12} [X]")
    
    print("\n" + "="*70)
    
    if accuracy >= 80:
        print("[OK] EXCELLENT: Fingerprinting technology is working!")
    elif accuracy >= 60:
        print("[!] GOOD: Significant improvement with fingerprinting")
    elif accuracy >= 40:
        print("[!] MODERATE: Some improvement, needs tuning")
    else:
        print("[X] POOR: Fingerprinting needs further refinement")
    
    print("="*70)

if __name__ == "__main__":
    test_ai_voices()
