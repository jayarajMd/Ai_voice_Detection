"""
Download Wav2Vec2 model properly with progress tracking
"""

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

print("="*80)
print("Downloading Wav2Vec2 Model (facebook/wav2vec2-base-960h)")
print("="*80)
print("\nThis is a 755MB download - may take a few minutes...")
print("Please keep your internet connection stable.\n")

try:
    # Download processor (small, fast)
    print("[1/2] Downloading processor...")
    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=None  # Use default cache
    )
    print("✓ Processor downloaded successfully\n")
    
    # Download model (large, takes time)
    print("[2/2] Downloading model (755MB)...")
    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h",
        cache_dir=None  # Use default cache
    )
    print("✓ Model downloaded successfully\n")
    
    # Test the model
    print("Testing model...")
    with torch.no_grad():
        dummy_input = torch.randn(1, 16000)
        output = model(dummy_input)
    print("✓ Model works correctly!\n")
    
    print("="*80)
    print("✅ SUCCESS: Wav2Vec2 model is ready to use")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Try again - downloads can resume from where they stopped")
    print("3. If it keeps failing, the system will work without deep learning")
