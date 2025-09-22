# from pathlib import Path
# import json
# import pandas as pd
# # from docx import Document
# import base64
# import fitz  # PyMuPDF for image-based PDF rendering
# import requests
# from io import BytesIO
# from PIL import Image
# from faster_whisper import WhisperModel

# # === Allowed File Types ===
# DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.json', '.xlsx'}
# IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
# AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac'}
# ALLOWED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS

# # === Whisper Model for Transcription ===
# audio_model = WhisperModel("large-v3", device="cuda", compute_type="float16")


# # === Document Extraction ===
# def handle_document(path: Path) -> dict:
#     try:
#         if path.suffix == '.txt':
#             text = path.read_text()
#         elif path.suffix == '.csv':
#             df = pd.read_csv(path)
#             text = df.to_string()
#         elif path.suffix == '.json':
#             with open(path, 'r', encoding='utf-8') as f:
#                 json_data = json.load(f)
#             text = json.dumps(json_data, indent=2)
#         elif path.suffix == '.xlsx':
#             df = pd.read_excel(path)
#             text = df.to_string()
#         else:
#             text = "[Unsupported document type]"
#     except Exception as e:
#         text = f"[Error reading document: {str(e)}]"

#     return {
#         "type": path.suffix[1:],
#         "content": text.strip(),
#         "file_path": str(path)
#     }

# # === PDF Handler with Inline Analysis ===
# def handle_pdf_with_fallback(path: Path) -> dict:
#     try:
#         doc = fitz.open(path)
#         output_chunks = []

#         for page_num, page in enumerate(doc):
#             blocks = page.get_text("dict")["blocks"]
#             output_chunks.append(f"=== Page {page_num + 1} ===\n")
#             for block in blocks:
#                 if block["type"] == 0:  # Text block
#                     for line in block["lines"]:
#                         line_text = "".join(span["text"] for span in line["spans"])
#                         output_chunks.append(line_text + "\n")
#                 elif block["type"] == 1:  # Image block
#                     images = page.get_images(full=True)
#                     for img in images:
#                         xref = img[0]
#                         base_image = doc.extract_image(xref)
#                         image_bytes = base_image["image"]
#                         image_ext = base_image["ext"]
#                         label = f"Page {page_num + 1}, Image {xref}"
#                         analysis = handle_image_from_bytes(image_bytes, label)
#                         output_chunks.append(f"[Image Analysis – {label}]:\n{analysis['result']}\n")

#         merged = "\n".join(output_chunks)
#         return {
#             "type": "pdf",
#             "content": merged.strip(),
#             "file_path": str(path)
#         }

#     except Exception as e:
#         return {
#             "type": "pdf",
#             "content": f"[Error: {str(e)}]",
#             "file_path": str(path)
#         }

# # === DOCX Handler with Proper Image Extraction ===
# def handle_docx_with_fallback(path: Path) -> dict:
#     try:
#         doc = Document(str(path))
#         output_chunks = ["=== DOCX Document ===\n"]

#         for para in doc.paragraphs:
#             output_chunks.append(para.text.strip())

#         for i, table in enumerate(doc.tables):
#             output_chunks.append(f"\n[Table {i + 1}]:")
#             for row in table.rows:
#                 row_text = " | ".join(cell.text.strip() for cell in row.cells)
#                 output_chunks.append(row_text)

#         rels = doc.part._rels
#         img_count = 1
#         for rel in rels:
#             rel_obj = rels[rel]
#             if "image" in rel_obj.target_ref:
#                 image_bytes = rel_obj.target_part.blob
#                 label = f"Image {img_count}"
#                 analysis = handle_image_from_bytes(image_bytes, label)
#                 output_chunks.append(f"[Image Analysis – {label}]:\n{analysis['result']}\n")
#                 img_count += 1

#         merged = "\n".join(output_chunks)
#         return {
#             "type": "docx",
#             "content": merged.strip(),
#             "file_path": str(path)
#         }

#     except Exception as e:
#         return {
#             "type": "docx",
#             "content": f"[Error reading DOCX: {str(e)}]",
#             "file_path": str(path)
#         }

# # === Image Analysis ===
# def handle_image(path: Path) -> dict:
#     try:
#         with open(path, "rb") as image_file:
#             image_bytes = image_file.read()
#         return handle_image_from_bytes(image_bytes, str(path))
#     except Exception as e:
#         return {
#             "type": "image",
#             "content": f"[Error during visual LLM analysis: {str(e)}]",
#             "file_path": str(path)
#         }

# def handle_image_from_bytes(image_bytes: bytes, label: str) -> dict:
#     try:
#         image_base64 = base64.b64encode(image_bytes).decode("utf-8")

#         prompt = (
#             "You are a visual understanding agent. Carefully analyze the content of this image. "
#             "Extract all readable text, recognize layout elements, and describe what the image represents. "
#             "If it's a document, extract its key fields. If it's a screenshot or graphic, summarize its purpose."
#         )

#         payload = {
#             "model": "qwen2.5vl:7b",
#             "prompt": prompt,
#             "images": [image_base64],
#             "stream": False
#         }

#         response = requests.post("http://3.11.3.161:11434/api/generate", json=payload)
#         result = response.json().get("response", "[No response returned]")

#     except Exception as e:
#         result = f"[Error during visual LLM analysis: {str(e)}]"

#     return {
#         "label": label,
#         "result": result.strip()
#     }

# # === Audio Handler ===
# def handle_audio(path: Path) -> dict:
#     try:
#         segments, info = audio_model.transcribe(str(path), beam_size=5)
#         transcript = "\n".join([f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}" for segment in segments])
#         lang_info = f"Detected language '{info.language}' with probability {info.language_probability:.2f}\n"
#         full_text = lang_info + transcript
#     except Exception as e:
#         full_text = f"[Error transcribing audio: {str(e)}]"

#     return {
#         "type": "audio",
#         "content": full_text.strip(),
#         "file_path": str(path)
#     }