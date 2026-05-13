from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os
import torch

model_path = "models/distilbert_emotion"

print(f"📂 Checking folder: {model_path}")
print(f"Path exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    print(f"\n📁 Files found:")
    for f in os.listdir(model_path):
        print(f"   - {f}")

print("\n🔄 Attempting to load model...")

try:
    # Try to load model from folder (should work with safetensors)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    print("✅ Model loaded successfully!")
    
    # Save in pytorch format (creates pytorch_model.bin)
    model.save_pretrained(model_path)
    print("✅ Saved as pytorch_model.bin")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n💡 Trying alternative method...")
    
    # Alternative: Use a specific config
    from transformers import AutoConfig, AutoModelForSequenceClassification
    
    print("Loading config from local files...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        from_tf=False,
        config=AutoConfig.from_pretrained(model_path)
    )
    print("✅ Model loaded via AutoModel!")