import requests
import base64

OLLAMA_URL = "http://3.11.3.161:11434/api/generate"
MODEL = "qwen2.5vl:7b"

def load_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def chat_with_image(image_path, prompt):
    image_base64 = load_image_as_base64(image_path)

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    result = response.json()
    print("\n🤖 Qwen2.5VL:", result["response"])

if __name__ == "__main__":
    print("🤖 Qwen2.5VL Vision Chat")
    image = input("Path to image file: ").strip()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        chat_with_image(image, user_input)