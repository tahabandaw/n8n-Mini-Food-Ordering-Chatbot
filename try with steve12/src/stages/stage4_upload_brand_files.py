from pathlib import Path

# Supported file types
DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.json', '.xlsx'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac'}
ALLOWED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS

def is_allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def save_brand_file_paths(file_paths: list):
    saved_paths = []

    for file_path in file_paths:
        file = Path(file_path)
        if not file.exists():
            print(f"⚠️ File not found: {file}")
            continue
        if not is_allowed_file(file.name):
            print(f"⚠️ Unsupported file type: {file.name}")
            continue

        saved_paths.append(str(file.resolve()))
        print(f"✅ Path accepted: {file.name}")

    print(f"\n🎯 {len(saved_paths)} valid file paths recorded.")
    return saved_paths