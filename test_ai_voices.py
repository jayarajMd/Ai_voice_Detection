"""
Test AI Voice Detection System on Real AI-Generated Voice Samples

This script tests all audio files in the ai_voices folder and generates
a comprehensive report with detection results.
"""

import logging
from pathlib import Path
from ai_voice_detector import AIVoiceDetector, AudioConfig
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ai_voices_folder():
    """Test all audio files in the ai_voices folder"""
    
    # Path to the ai_voices folder
    ai_voices_dir = Path(r"c:\Users\mugam\Downloads\ai_voices")
    
    if not ai_voices_dir.exists():
        logger.error(f"Folder not found: {ai_voices_dir}")
        return
    
    # Get all audio files
    audio_files = list(ai_voices_dir.glob("*.mp3")) + \
                  list(ai_voices_dir.glob("*.wav")) + \
                  list(ai_voices_dir.glob("*.m4a"))
    
    if not audio_files:
        logger.error(f"No audio files found in {ai_voices_dir}")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING AI VOICE DETECTION SYSTEM")
    logger.info(f"{'='*80}")
    logger.info(f"Found {len(audio_files)} audio files to test")
    logger.info(f"")
    
    # Initialize detector
    config = AudioConfig(
        target_sr=16000,
        ai_threshold=0.6  # Standard threshold
    )
    
    # Set logging to WARNING to reduce noise
    logging.getLogger('ai_voice_detector').setLevel(logging.WARNING)
    
    detector = AIVoiceDetector(config)
    
    # Store results
    results = []
    correct_detections = 0
    
    # Test each file
    for i, audio_file in enumerate(sorted(audio_files), 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST {i}/{len(audio_files)}: {audio_file.name}")
        logger.info(f"{'='*80}")
        
        try:
            # Run detection
            result = detector.detect(str(audio_file))
            
            # Extract key information
            file_info = {
                'filename': audio_file.name,
                'language': audio_file.stem.split('_')[1],  # Extract language from filename
                'sample_num': audio_file.stem.split('_')[2],  # Extract sample number
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'final_score': result['final_score'],
                'individual_scores': {
                    'pitch_stability': result['test_results']['pitch_stability'],
                    'spectral_smoothness': result['test_results']['spectral_smoothness'],
                    'phase_artifacts': result['test_results']['phase_artifacts'],
                    'noise_randomness': result['test_results']['noise_randomness'],
                }
            }
            
            results.append(file_info)
            
            # Display results
            logger.info(f"\n📊 DETECTION RESULTS:")
            logger.info(f"   Language:     {file_info['language'].upper()}")
            logger.info(f"   Sample:       #{file_info['sample_num']}")
            logger.info(f"   Prediction:   {file_info['prediction']}")
            logger.info(f"   Confidence:   {file_info['confidence']:.1f}%")
            logger.info(f"   Final Score:  {file_info['final_score']:.4f}")
            logger.info(f"")
            logger.info(f"   Individual Test Scores:")
            logger.info(f"      Pitch Stability:     {file_info['individual_scores']['pitch_stability']:.3f}")
            logger.info(f"      Spectral Smoothness: {file_info['individual_scores']['spectral_smoothness']:.3f}")
            logger.info(f"      Phase Artifacts:     {file_info['individual_scores']['phase_artifacts']:.3f}")
            logger.info(f"      Noise Randomness:    {file_info['individual_scores']['noise_randomness']:.3f}")
            
            # Check if correct (these are all AI-generated)
            if file_info['prediction'] == 'AI_GENERATED':
                logger.info(f"\n   ✅ CORRECT: Identified as AI-generated")
                correct_detections += 1
            else:
                logger.info(f"\n   ❌ INCORRECT: Misclassified as HUMAN")
            
        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}")
            logger.exception(e)
    
    # Generate summary report
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total Files Tested:     {len(results)}")
    logger.info(f"Correct Detections:     {correct_detections}")
    logger.info(f"Incorrect Detections:   {len(results) - correct_detections}")
    logger.info(f"Accuracy:               {correct_detections/len(results)*100:.1f}%")
    logger.info(f"")
    
    # Group by language
    logger.info(f"\n📊 RESULTS BY LANGUAGE:")
    logger.info(f"{'='*80}")
    
    languages = {}
    for r in results:
        lang = r['language']
        if lang not in languages:
            languages[lang] = {'correct': 0, 'total': 0, 'scores': []}
        
        languages[lang]['total'] += 1
        if r['prediction'] == 'AI_GENERATED':
            languages[lang]['correct'] += 1
        languages[lang]['scores'].append(r['final_score'])
    
    for lang, stats in sorted(languages.items()):
        accuracy = stats['correct'] / stats['total'] * 100
        avg_score = sum(stats['scores']) / len(stats['scores'])
        logger.info(f"\n{lang.upper()}:")
        logger.info(f"   Samples:          {stats['total']}")
        logger.info(f"   Correct:          {stats['correct']}/{stats['total']}")
        logger.info(f"   Accuracy:         {accuracy:.1f}%")
        logger.info(f"   Avg Final Score:  {avg_score:.4f}")
    
    # Confidence distribution
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 CONFIDENCE DISTRIBUTION:")
    logger.info(f"{'='*80}")
    
    high_conf = sum(1 for r in results if r['confidence'] >= 70)
    medium_conf = sum(1 for r in results if 50 <= r['confidence'] < 70)
    low_conf = sum(1 for r in results if r['confidence'] < 50)
    
    logger.info(f"High Confidence (≥70%):    {high_conf} files")
    logger.info(f"Medium Confidence (50-70%): {medium_conf} files")
    logger.info(f"Low Confidence (<50%):      {low_conf} files")
    
    # Save detailed report to JSON
    report_file = Path("ai_voices_test_report.json")
    report = {
        'test_date': datetime.now().isoformat(),
        'total_files': len(results),
        'correct_detections': correct_detections,
        'accuracy': correct_detections / len(results) * 100,
        'threshold_used': config.ai_threshold,
        'results': results,
        'by_language': languages
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📄 Detailed report saved to: {report_file}")
    logger.info(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    test_ai_voices_folder()
