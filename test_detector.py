"""
Test Script for AI Voice Detection System
==========================================

This script provides comprehensive testing and validation of the detection system.

Usage:
    python test_detector.py
"""

import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
import logging

# Import the detector
from ai_voice_detector import AIVoiceDetector, AudioConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestAudioGenerator:
    """Generate synthetic test audio samples."""
    
    @staticmethod
    def generate_sine_wave(duration=5.0, frequency=440, sr=16000):
        """Generate pure sine wave (simulates AI-like tone)."""
        t = np.linspace(0, duration, int(duration * sr))
        audio = np.sin(2 * np.pi * frequency * t)
        return audio.astype(np.float32)
    
    @staticmethod
    def generate_complex_wave(duration=5.0, sr=16000):
        """Generate complex harmonic wave (more human-like)."""
        t = np.linspace(0, duration, int(duration * sr))
        
        # Fundamental + harmonics
        audio = (
            np.sin(2 * np.pi * 200 * t) +  # Fundamental
            0.5 * np.sin(2 * np.pi * 400 * t) +  # 2nd harmonic
            0.25 * np.sin(2 * np.pi * 600 * t) +  # 3rd harmonic
            0.1 * np.random.randn(len(t))  # Noise
        )
        
        # Add natural pitch variation
        pitch_modulation = 1 + 0.02 * np.sin(2 * np.pi * 3 * t)
        audio = audio * pitch_modulation
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        return audio.astype(np.float32)
    
    @staticmethod
    def generate_white_noise(duration=5.0, sr=16000):
        """Generate white noise."""
        audio = np.random.randn(int(duration * sr))
        audio = audio / np.max(np.abs(audio)) * 0.5
        return audio.astype(np.float32)
    
    @staticmethod
    def save_audio(audio, filename, sr=16000):
        """Save audio to file."""
        # Scale to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(filename, sr, audio_int16)
        logger.info(f"Saved test audio: {filename}")


def test_detector_basic():
    """Test basic detector functionality."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Basic Functionality")
    logger.info("="*80)
    
    # Create test directory
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    
    # Generate test samples
    generator = TestAudioGenerator()
    
    # Sample 1: Pure sine (AI-like)
    audio_sine = generator.generate_sine_wave()
    sine_path = test_dir / "test_sine.wav"
    generator.save_audio(audio_sine, sine_path)
    
    # Sample 2: Complex wave (human-like)
    audio_complex = generator.generate_complex_wave()
    complex_path = test_dir / "test_complex.wav"
    generator.save_audio(audio_complex, complex_path)
    
    # Sample 3: White noise
    audio_noise = generator.generate_white_noise()
    noise_path = test_dir / "test_noise.wav"
    generator.save_audio(audio_noise, noise_path)
    
    # Initialize detector
    detector = AIVoiceDetector()
    
    # Test each sample
    samples = [
        ("Sine Wave (AI-like)", sine_path, "AI_GENERATED"),
        ("Complex Wave (Human-like)", complex_path, "HUMAN"),
        ("White Noise", noise_path, None)  # Uncertain
    ]
    
    results = []
    for name, path, expected in samples:
        logger.info(f"\nTesting: {name}")
        logger.info("-" * 60)
        
        try:
            result = detector.detect(str(path))
            
            logger.info(f"✓ Result: {result['label']}")
            logger.info(f"  Confidence: {result['confidence']:.2%}")
            logger.info(f"  Final Score: {result['final_score']:.4f}")
            
            results.append({
                'name': name,
                'expected': expected,
                'actual': result['label'],
                'confidence': result['confidence'],
                'passed': expected is None or result['label'] == expected
            })
            
        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            results.append({
                'name': name,
                'expected': expected,
                'actual': 'ERROR',
                'confidence': 0.0,
                'passed': False
            })
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        logger.info(f"{status} - {r['name']}")
        logger.info(f"       Expected: {r['expected']}, Got: {r['actual']}")
    
    passed = sum(1 for r in results if r['passed'])
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    return results


def test_detector_edge_cases():
    """Test edge cases and error handling."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Edge Cases")
    logger.info("="*80)
    
    detector = AIVoiceDetector()
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    generator = TestAudioGenerator()
    
    edge_cases = []
    
    # Case 1: Very short audio (0.5s)
    logger.info("\n1. Testing very short audio (0.5s)...")
    audio_short = generator.generate_complex_wave(duration=0.5)
    short_path = test_dir / "test_short.wav"
    generator.save_audio(audio_short, short_path)
    
    try:
        result = detector.detect(str(short_path))
        logger.info(f"✓ Short audio processed: {result['label']}")
        edge_cases.append(('Short audio', True))
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        edge_cases.append(('Short audio', False))
    
    # Case 2: Silent audio
    logger.info("\n2. Testing silent audio...")
    audio_silent = np.zeros(16000 * 3, dtype=np.float32)  # 3 seconds of silence
    silent_path = test_dir / "test_silent.wav"
    generator.save_audio(audio_silent, silent_path)
    
    try:
        result = detector.detect(str(silent_path))
        logger.info(f"✓ Silent audio processed: {result['label']}")
        edge_cases.append(('Silent audio', True))
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        edge_cases.append(('Silent audio', False))
    
    # Case 3: Invalid file path
    logger.info("\n3. Testing invalid file path...")
    try:
        result = detector.detect("nonexistent_file.wav")
        logger.error("✗ Should have raised FileNotFoundError")
        edge_cases.append(('Invalid path', False))
    except FileNotFoundError:
        logger.info("✓ Correctly raised FileNotFoundError")
        edge_cases.append(('Invalid path', True))
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        edge_cases.append(('Invalid path', False))
    
    # Case 4: Invalid file format (if we create a .txt file)
    logger.info("\n4. Testing invalid file format...")
    txt_path = test_dir / "test_invalid.txt"
    with open(txt_path, 'w') as f:
        f.write("This is not audio")
    
    try:
        result = detector.detect(str(txt_path))
        logger.error("✗ Should have raised ValueError")
        edge_cases.append(('Invalid format', False))
    except ValueError:
        logger.info("✓ Correctly raised ValueError")
        edge_cases.append(('Invalid format', True))
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        edge_cases.append(('Invalid format', False))
    
    # Summary
    logger.info("\n" + "-"*80)
    passed = sum(1 for _, success in edge_cases if success)
    logger.info(f"Edge Cases: {passed}/{len(edge_cases)} passed")
    
    return edge_cases


def test_detector_configuration():
    """Test custom configuration."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Configuration")
    logger.info("="*80)
    
    test_dir = Path("test_audio")
    generator = TestAudioGenerator()
    
    # Generate test audio
    audio = generator.generate_complex_wave()
    audio_path = test_dir / "test_config.wav"
    generator.save_audio(audio, audio_path)
    
    # Test different thresholds
    thresholds = [0.3, 0.5, 0.6, 0.7, 0.9]
    results = []
    
    for threshold in thresholds:
        config = AudioConfig(ai_threshold=threshold)
        detector = AIVoiceDetector(config)
        
        result = detector.detect(str(audio_path))
        
        logger.info(f"\nThreshold: {threshold:.1f}")
        logger.info(f"  Label: {result['label']}")
        logger.info(f"  Score: {result['final_score']:.4f}")
        
        results.append({
            'threshold': threshold,
            'label': result['label'],
            'score': result['final_score']
        })
    
    # Verify threshold logic
    logger.info("\n" + "-"*80)
    logger.info("Threshold Logic Verification:")
    
    score = results[0]['score']  # Same audio, same score
    for r in results:
        expected_label = "AI_GENERATED" if score > r['threshold'] else "HUMAN"
        actual_label = r['label']
        status = "✓" if expected_label == actual_label else "✗"
        logger.info(f"{status} Threshold {r['threshold']:.1f}: {actual_label} (expected {expected_label})")
    
    return results


def test_detector_consistency():
    """Test consistency of results."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Consistency")
    logger.info("="*80)
    
    test_dir = Path("test_audio")
    generator = TestAudioGenerator()
    
    # Generate test audio
    audio = generator.generate_complex_wave()
    audio_path = test_dir / "test_consistency.wav"
    generator.save_audio(audio, audio_path)
    
    # Run detector multiple times
    detector = AIVoiceDetector()
    runs = 3
    results = []
    
    logger.info(f"\nRunning detector {runs} times on same audio...\n")
    
    for i in range(runs):
        result = detector.detect(str(audio_path))
        results.append(result)
        logger.info(f"Run {i+1}: {result['label']} (score: {result['final_score']:.4f})")
    
    # Check consistency
    labels = [r['label'] for r in results]
    scores = [r['final_score'] for r in results]
    
    all_same_label = len(set(labels)) == 1
    score_variance = np.var(scores)
    
    logger.info("\n" + "-"*80)
    if all_same_label:
        logger.info("✓ Labels are consistent")
    else:
        logger.error("✗ Labels are inconsistent!")
    
    logger.info(f"Score variance: {score_variance:.8f}")
    if score_variance < 1e-6:
        logger.info("✓ Scores are deterministic")
    else:
        logger.warning("⚠ Scores have slight variance (may be acceptable)")
    
    return all_same_label and score_variance < 1e-4


def test_individual_components():
    """Test individual pipeline components."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Component Testing")
    logger.info("="*80)
    
    from ai_voice_detector import (
        AudioInputHandler, AudioPreprocessor, FeatureExtractor,
        DetectionTests, ScoreFusion, Classifier
    )
    
    test_dir = Path("test_audio")
    generator = TestAudioGenerator()
    
    # Generate test audio
    audio = generator.generate_complex_wave()
    audio_path = test_dir / "test_components.wav"
    generator.save_audio(audio, audio_path)
    
    config = AudioConfig()
    results = []
    
    # Test 1: Input Handler
    logger.info("\n1. Testing Input Handler...")
    try:
        handler = AudioInputHandler(config)
        audio_data, sr = handler.load_audio(str(audio_path))
        logger.info(f"✓ Loaded audio: shape {audio_data.shape}, SR {sr}Hz")
        results.append(('Input Handler', True))
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        results.append(('Input Handler', False))
        return results
    
    # Test 2: Preprocessor
    logger.info("\n2. Testing Preprocessor...")
    try:
        preprocessor = AudioPreprocessor(config)
        audio_proc, stats = preprocessor.process(audio_data, sr)
        logger.info(f"✓ Preprocessed: {len(audio_proc)} samples")
        logger.info(f"  Stats: {list(stats.keys())}")
        results.append(('Preprocessor', True))
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        results.append(('Preprocessor', False))
        return results
    
    # Test 3: Feature Extractor
    logger.info("\n3. Testing Feature Extractor...")
    try:
        extractor = FeatureExtractor(config)
        features = extractor.extract_all(audio_proc, config.target_sr)
        logger.info(f"✓ Extracted features: {list(features.keys())}")
        results.append(('Feature Extractor', True))
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        results.append(('Feature Extractor', False))
        return results
    
    # Test 4: Detection Tests
    logger.info("\n4. Testing Detection Tests...")
    try:
        tests = DetectionTests(config)
        test_results = tests.run_all_tests(features)
        logger.info(f"✓ Ran tests: {list(test_results.keys())}")
        results.append(('Detection Tests', True))
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        results.append(('Detection Tests', False))
        return results
    
    # Test 5: Score Fusion
    logger.info("\n5. Testing Score Fusion...")
    try:
        fusion = ScoreFusion(config)
        fused = fusion.fuse_scores(test_results)
        logger.info(f"✓ Fused score: {fused['final_score']:.4f}")
        results.append(('Score Fusion', True))
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        results.append(('Score Fusion', False))
        return results
    
    # Test 6: Classifier
    logger.info("\n6. Testing Classifier...")
    try:
        classifier = Classifier(config)
        classification = classifier.classify(fused)
        logger.info(f"✓ Classification: {classification['label']}")
        results.append(('Classifier', True))
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        results.append(('Classifier', False))
        return results
    
    # Summary
    logger.info("\n" + "-"*80)
    passed = sum(1 for _, success in results if success)
    logger.info(f"Components: {passed}/{len(results)} passed")
    
    return results


def run_all_tests():
    """Run complete test suite."""
    logger.info("\n" + "="*80)
    logger.info("AI VOICE DETECTION SYSTEM - TEST SUITE")
    logger.info("="*80)
    
    all_results = []
    
    try:
        # Test 1: Basic functionality
        basic_results = test_detector_basic()
        all_results.append(('Basic Functionality', basic_results))
        
        # Test 2: Edge cases
        edge_results = test_detector_edge_cases()
        all_results.append(('Edge Cases', edge_results))
        
        # Test 3: Configuration
        config_results = test_detector_configuration()
        all_results.append(('Configuration', config_results))
        
        # Test 4: Consistency
        consistency_passed = test_detector_consistency()
        all_results.append(('Consistency', consistency_passed))
        
        # Test 5: Components
        component_results = test_individual_components()
        all_results.append(('Components', component_results))
        
    except Exception as e:
        logger.error(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("FINAL TEST SUMMARY")
    logger.info("="*80)
    
    for test_name, results in all_results:
        if isinstance(results, bool):
            status = "✓ PASS" if results else "✗ FAIL"
            logger.info(f"{status} - {test_name}")
        elif isinstance(results, list) and results and isinstance(results[0], dict):
            if 'passed' in results[0]:
                passed = sum(1 for r in results if r.get('passed', False))
                logger.info(f"  {test_name}: {passed}/{len(results)} passed")
            else:
                passed = sum(1 for _, success in results if success)
                logger.info(f"  {test_name}: {passed}/{len(results)} passed")
        else:
            logger.info(f"  {test_name}: Complete")
    
    logger.info("\n✓ All tests completed")
    logger.info("="*80)


if __name__ == "__main__":
    run_all_tests()
