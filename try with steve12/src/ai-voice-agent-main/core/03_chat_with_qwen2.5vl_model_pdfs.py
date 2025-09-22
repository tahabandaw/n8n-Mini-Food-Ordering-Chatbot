import base64
import requests
from pdf2image import convert_from_path
import io

MODEL = "qwen2.5vl:7b"
OLLAMA_URL = "http://3.11.3.161:11434/api/generate"

def convert_pdf_to_images(pdf_path):
    pages = convert_from_path(pdf_path)
    return pages

def image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def send_to_qwen2_5vl(prompt, base64_image):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }
    res = requests.post(OLLAMA_URL, json=payload)
    res.raise_for_status()
    return res.json()["response"]

def process_pdf_with_prompt(pdf_path, prompt):
    pages = convert_pdf_to_images(pdf_path)
    print(f"📄 PDF has {len(pages)} pages. Sending to Qwen2.5VL...\n")

    for i, page_img in enumerate(pages, 1):
        print(f"--- Page {i} ---")
        base64_img = image_to_base64(page_img)
        response = send_to_qwen2_5vl(prompt, base64_img)
        print(response)
        print()

if __name__ == "__main__":
    pdf_file = input("Enter path to PDF file: ").strip()
    user_prompt = input("Enter your prompt (e.g. summarize, extract info): ").strip()
    process_pdf_with_prompt(pdf_file, user_prompt)