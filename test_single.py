#!/usr/bin/env python3
"""
Test a single AI voice file with fingerprinting system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ai_voice_detector import AIVoiceDetector

def test_single_file(filepath):
    """Test a single audio file."""
    
    print("="*80)
    print("AI VOICE DETECTION WITH ADVANCED FINGERPRINTING")
    print("="*80)
    print(f"\nTesting: {filepath}\n")
    print("9 Detection Methods:")
    print("  Traditional (60%):")
    print("    - Pitch Stability (8%)")
    print("    - Spectral Smoothness (10%)")
    print("    - Phase Artifacts (10%)")
    print("    - Noise Randomness (12%)")
    print("    - Deep Learning (20%)")
    print("  Fingerprinting (40%):")
    print("    - Spectral Fingerprint (15%) - AI frequency signatures")
    print("    - Temporal Fingerprint (10%) - Unnatural timing patterns")
    print("    - Micro-Artifacts (8%) - Neural synthesis glitches")
    print("    - Prosody Fingerprint (7%) - Controlled rhythm/intonation")
    print("\n" + "="*80)
    
    detector = AIVoiceDetector()
    result = detector.detect(filepath)
    
    print("\n" + "="*80)
    print("FINAL RESULT")
    print("="*80)
    print(f"\nFinal Score: {result['final_score']:.4f}")
    print(f"Classification: {result['label']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"\nDetailed Scores:")
    for test_name, score in result['individual_scores'].items():
        print(f"  {test_name}: {score:.3f}")
    print("\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_single_file(sys.argv[1])
    else:
        test_single_file("sample voice 1.mp3")
