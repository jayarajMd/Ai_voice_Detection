"""
Complete Example Usage - AI Voice Detection System
==================================================

This script demonstrates all features of the AI voice detection system.
"""

from pathlib import Path
import json
from ai_voice_detector import AIVoiceDetector, AudioConfig


def example_1_basic_detection():
    """Example 1: Basic detection on a single file."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Detection")
    print("="*80)
    
    # Initialize detector with default settings
    detector = AIVoiceDetector()
    
    # Detect (replace with your audio file)
    audio_path = "test_audio/test_complex.wav"
    
    print(f"\nAnalyzing: {audio_path}")
    
    result = detector.detect(audio_path)
    
    # Print results
    print("\n" + "-"*80)
    print("RESULT:")
    print(f"  Label:      {result['label']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Score:      {result['final_score']:.4f}")
    print(f"  Certainty:  {result['certainty']}")


def example_2_custom_config():
    """Example 2: Using custom configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Configuration")
    print("="*80)
    
    # Create custom configuration
    config = AudioConfig(
        max_duration_sec=20.0,      # Allow longer audio
        ai_threshold=0.5,            # More sensitive (lower threshold)
        weight_ml=0.50,              # Trust ML model more
        weight_pitch=0.10,           # Less weight on pitch
        weight_spectral=0.15,
        weight_phase=0.15,
        weight_noise=0.10,
        trim_db=25,                  # More aggressive silence trimming
        denoise=True                 # Enable denoising
    )
    
    print("\nCustom Configuration:")
    print(f"  AI Threshold: {config.ai_threshold}")
    print(f"  Max Duration: {config.max_duration_sec}s")
    print(f"  ML Weight: {config.weight_ml:.0%}")
    print(f"  Denoising: {config.denoise}")
    
    # Initialize with custom config
    detector = AIVoiceDetector(config)
    
    # Detect
    audio_path = "test_audio/test_complex.wav"
    result = detector.detect(audio_path)
    
    print(f"\nResult: {result['label']} ({result['confidence']:.1%})")


def example_3_detailed_analysis():
    """Example 3: Accessing detailed analysis results."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Detailed Analysis")
    print("="*80)
    
    detector = AIVoiceDetector()
    audio_path = "test_audio/test_complex.wav"
    
    result = detector.detect(audio_path)
    
    # Print detailed breakdown
    print("\n📊 INDIVIDUAL TEST SCORES")
    print("-"*80)
    
    for test_name, score in result['individual_scores'].items():
        if score is not None:
            # Create visual bar
            bar_length = int(score * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            # Color coding (for terminals that support it)
            if score > 0.7:
                indicator = "🔴"  # High AI probability
            elif score > 0.4:
                indicator = "🟡"  # Medium
            else:
                indicator = "🟢"  # Low AI probability
            
            print(f"{indicator} {test_name:20s}: {score:.3f} [{bar}]")
        else:
            print(f"⚪ {test_name:20s}: N/A")
    
    print("\n📈 PREPROCESSING STATISTICS")
    print("-"*80)
    for key, value in result['preprocessing_stats'].items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.4f}")
        else:
            print(f"  {key:25s}: {value}")
    
    print("\n🔬 DETAILED TEST METRICS")
    print("-"*80)
    
    # Pitch test details
    if 'pitch_stability' in result['test_details']:
        pitch = result['test_details']['pitch_stability']
        print("\nPitch Stability:")
        print(f"  Jitter:        {pitch['jitter']:.6f}")
        print(f"  Shimmer:       {pitch['shimmer']:.6f}")
        print(f"  Pitch Var:     {pitch['pitch_variance']:.2f}")
    
    # Spectral test details
    if 'spectral_smoothness' in result['test_details']:
        spectral = result['test_details']['spectral_smoothness']
        print("\nSpectral Smoothness:")
        print(f"  Entropy:       {spectral['spectral_entropy']:.4f}")
        print(f"  Harmonic Var:  {spectral['harmonic_variance']:.2e}")
        print(f"  Flatness:      {spectral['flatness_mean']:.4f}")
    
    # Phase test details
    if 'phase_artifacts' in result['test_details']:
        phase = result['test_details']['phase_artifacts']
        print("\nPhase Artifacts:")
        print(f"  Coherence:     {phase['phase_coherence']:.4f}")
        print(f"  GD Variance:   {phase['group_delay_variance']:.6f}")
    
    # Noise test details
    if 'noise_randomness' in result['test_details']:
        noise = result['test_details']['noise_randomness']
        print("\nNoise Randomness:")
        print(f"  Autocorr:      {noise['residual_autocorr']:.4f}")
        print(f"  Entropy:       {noise['residual_entropy']:.4f}")
        print(f"  HNR:           {noise['hnr']:.2f} dB")
    
    # ML test details
    if 'deep_learning' in result['test_details']:
        ml = result['test_details']['deep_learning']
        if ml['embedding_variance'] is not None:
            print("\nDeep Learning:")
            print(f"  Embed Var:     {ml['embedding_variance']:.6f}")
            print(f"  Temporal Cons: {ml['temporal_consistency']:.4f}")


def example_4_batch_processing():
    """Example 4: Process multiple files."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Processing")
    print("="*80)
    
    detector = AIVoiceDetector()
    
    # Get all audio files in test directory
    audio_dir = Path("test_audio")
    audio_files = list(audio_dir.glob("*.wav"))
    
    if not audio_files:
        print("\nNo audio files found in test_audio/")
        print("Run test_detector.py first to generate test samples.")
        return
    
    print(f"\nProcessing {len(audio_files)} files...")
    
    results = {}
    
    for audio_file in audio_files:
        print(f"\n  Processing: {audio_file.name}")
        
        try:
            result = detector.detect(str(audio_file))
            results[audio_file.name] = {
                'label': result['label'],
                'confidence': result['confidence'],
                'score': result['final_score'],
                'certainty': result['certainty']
            }
            print(f"    → {result['label']} ({result['confidence']:.0%})")
            
        except Exception as e:
            print(f"    → ERROR: {e}")
            results[audio_file.name] = {'error': str(e)}
    
    # Summary
    print("\n" + "-"*80)
    print("SUMMARY")
    print("-"*80)
    
    ai_count = sum(1 for r in results.values() if r.get('label') == 'AI_GENERATED')
    human_count = sum(1 for r in results.values() if r.get('label') == 'HUMAN')
    error_count = sum(1 for r in results.values() if 'error' in r)
    
    print(f"\n  Total files:     {len(results)}")
    print(f"  AI-generated:    {ai_count}")
    print(f"  Human:           {human_count}")
    print(f"  Errors:          {error_count}")
    
    # Detailed table
    print("\n  Detailed Results:")
    print(f"  {'Filename':<30} {'Label':<15} {'Confidence':<12} {'Certainty'}")
    print("  " + "-"*76)
    
    for filename, result in sorted(results.items()):
        if 'error' in result:
            print(f"  {filename:<30} {'ERROR':<15} {'-':<12} {'-'}")
        else:
            conf_str = f"{result['confidence']:.1%}"
            print(f"  {filename:<30} {result['label']:<15} {conf_str:<12} {result['certainty']}")


def example_5_export_results():
    """Example 5: Export results to JSON."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Export Results to JSON")
    print("="*80)
    
    detector = AIVoiceDetector()
    audio_path = "test_audio/test_complex.wav"
    
    print(f"\nAnalyzing: {audio_path}")
    result = detector.detect(audio_path)
    
    # Prepare for JSON export (convert numpy types)
    export_data = {
        'audio_file': audio_path,
        'classification': {
            'label': result['label'],
            'confidence': float(result['confidence']),
            'final_score': float(result['final_score']),
            'certainty': result['certainty']
        },
        'individual_scores': {
            k: float(v) if v is not None else None
            for k, v in result['individual_scores'].items()
        },
        'preprocessing_stats': {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in result['preprocessing_stats'].items()
        }
    }
    
    # Save to JSON
    output_file = "detection_result.json"
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n✓ Results exported to: {output_file}")
    
    # Print JSON
    print("\nJSON Content:")
    print(json.dumps(export_data, indent=2))


def example_6_compare_configurations():
    """Example 6: Compare different threshold settings."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Compare Configurations")
    print("="*80)
    
    audio_path = "test_audio/test_complex.wav"
    print(f"\nTesting with: {audio_path}")
    
    # Test different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print("\n" + "-"*80)
    print(f"{'Threshold':<12} {'Label':<15} {'Confidence':<12} {'Final Score'}")
    print("-"*80)
    
    for threshold in thresholds:
        config = AudioConfig(ai_threshold=threshold)
        detector = AIVoiceDetector(config)
        
        result = detector.detect(audio_path)
        
        conf_str = f"{result['confidence']:.1%}"
        score_str = f"{result['final_score']:.4f}"
        
        print(f"{threshold:<12.1f} {result['label']:<15} {conf_str:<12} {score_str}")
    
    print("\nNote: The final score remains constant; only the decision changes.")


def example_7_error_handling():
    """Example 7: Proper error handling."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Error Handling")
    print("="*80)
    
    detector = AIVoiceDetector()
    
    # Test cases
    test_cases = [
        ("nonexistent_file.wav", "File not found"),
        ("test_audio/test_invalid.txt", "Invalid format"),
        ("https://invalid-url.com/audio.wav", "URL error")
    ]
    
    for audio_path, expected_error in test_cases:
        print(f"\nTesting: {audio_path}")
        print(f"Expected: {expected_error}")
        
        try:
            result = detector.detect(audio_path)
            print(f"  Result: {result['label']}")
            
        except FileNotFoundError as e:
            print(f"  ✓ Caught FileNotFoundError: {e}")
            
        except ValueError as e:
            print(f"  ✓ Caught ValueError: {e}")
            
        except IOError as e:
            print(f"  ✓ Caught IOError: {e}")
            
        except Exception as e:
            print(f"  ⚠ Unexpected error: {type(e).__name__}: {e}")


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "="*80)
    print("AI VOICE DETECTION SYSTEM - COMPLETE EXAMPLES")
    print("="*80)
    
    examples = [
        ("Basic Detection", example_1_basic_detection),
        ("Custom Configuration", example_2_custom_config),
        ("Detailed Analysis", example_3_detailed_analysis),
        ("Batch Processing", example_4_batch_processing),
        ("Export Results", example_5_export_results),
        ("Compare Configurations", example_6_compare_configurations),
        ("Error Handling", example_7_error_handling)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            print(f"\n\n{'='*80}")
            print(f"Running Example {i}/{len(examples)}: {name}")
            print(f"{'='*80}")
            func()
        except Exception as e:
            print(f"\n❌ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Ask if user wants to continue
            response = input("\nContinue with next example? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("\n\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    # Check if test audio exists
    test_dir = Path("test_audio")
    if not test_dir.exists() or not list(test_dir.glob("*.wav")):
        print("\n⚠️  Warning: No test audio found!")
        print("Please run 'python test_detector.py' first to generate test samples.\n")
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Run all examples
    run_all_examples()
