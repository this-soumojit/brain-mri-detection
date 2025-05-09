import gdown
import os
import sys

# Configuration - CHANGE THESE!
MODEL_URL = "https://drive.google.com/uc?id=15jYYU8tf8NvAiQAdwdbLuBng5EBAT8ch"  # Your Drive link
MODEL_NAME = "brain_tumor_resnet50_20250415_1757.h5"  # Your actual model filename
MODEL_DIR = "downloaded_models"  # Folder to save models (separate from code)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    output_path = os.path.join(MODEL_DIR, MODEL_NAME)
    
    print(f"🔄 Downloading {MODEL_NAME} from Google Drive...")
    try:
        gdown.download(MODEL_URL, output_path, quiet=False)
        print(f"✅ Success! Model saved to:\n{os.path.abspath(output_path)}")
    except Exception as e:
        print(f"❌ Download failed: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()