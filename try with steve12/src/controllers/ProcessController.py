from .BaseController import BaseController
from .ProjectController import ProjectController
import os
# from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai import OpenAIEmbeddings

from models import ProcessingEnum

import os
import base64
import logging
from typing import Optional, List
from langchain.schema import Document
import requests
from PIL import Image
import io

# class ProcessController(BaseController):

#     def __init__(self, project_id: str):
#         super().__init__()

#         self.project_id = project_id
#         self.project_path = ProjectController().get_project_path(project_id=project_id)



    # def get_file_loader(self, file_id: str):

    #     file_ext = self.get_file_extension(file_id=file_id)
    #     file_path = os.path.join(
    #         self.project_path,
    #         file_id
    #     )

    #     if not os.path.exists(file_path):
    #         return None

    #     if file_ext == ProcessingEnum.TXT.value:
    #         return TextLoader(file_path, encoding="utf-8")

    #     if file_ext == ProcessingEnum.PDF.value:
    #         return PyMuPDFLoader(file_path)
        
    #     if file_ext == ProcessingEnum.DOCX.value:
    #         return Docx2txtLoader(file_path)
        
        
    #     return None

    # def get_file_content(self, file_id: str):

    #     loader = self.get_file_loader(file_id=file_id)
    #     if loader:
    #         return loader.load()

    #     return None




from langchain_community.document_loaders import (
    TextLoader, PyMuPDFLoader, Docx2txtLoader, 
    CSVLoader, JSONLoader, UnstructuredExcelLoader
)
# from pymupdf import PyMuPDF  # PyMuPDF for PDF handling
from faster_whisper import WhisperModel
from langchain.schema import Document

# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import json


# Supported file types
DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv', '.json', '.xlsx'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff','.opus'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac'}

class ProcessController(BaseController):
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_path = ProjectController().get_project_path(project_id=project_id)
        self.whisper_model = WhisperModel("small", device="cpu", compute_type="int8")  # Correct positional argument
        # self.image_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat")
        # self.image_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat")/

    def get_file_extension(self, file_id: str):
        return os.path.splitext(file_id)[-1]

    def get_file_loader(self, file_path: str, file_ext: str):
        loaders = {
            '.txt': lambda: TextLoader(file_path, encoding="utf-8"),
            '.pdf': lambda: PyMuPDFLoader(file_path),
            '.docx': lambda: Docx2txtLoader(file_path),
            '.csv': lambda: CSVLoader(file_path),
            '.json': lambda: JSONLoader(file_path),
            '.xlsx': lambda: UnstructuredExcelLoader(file_path)
        }
        return loaders.get(file_ext, lambda: None)()

    def process_audio(self, file_path: str):
        segments, info = self.whisper_model.transcribe(file_path, beam_size=5)
        # Combine all segments into one text
        full_text = " ".join([segment.text for segment in segments])
        return [Document(page_content=full_text)]

    # def process_image(self, file_path: str):
    #     image = Image.open(file_path)
    #     inputs = self.image_tokenizer.process_images(
    #         "Describe this image in detail", image
    #     )
    #     outputs = self.image_model.generate(**inputs)
    #     description = self.image_tokenizer.decode(outputs[0])
    #     return [Document(page_content=description)]

    def process_file(self, file_id: str):
        file_path = os.path.join(self.project_path, file_id)
        file_ext = self.get_file_extension(file_id)

        if not os.path.exists(file_path):
            return None

        if file_ext in DOCUMENT_EXTENSIONS:
            loader = self.get_file_loader(file_path, file_ext)
            return loader.load() if loader else None

        if file_ext in AUDIO_EXTENSIONS:
            return self.process_audio(file_path)

        if file_ext in IMAGE_EXTENSIONS:
            return self.handle_image(file_path)

        return None
    
    def process_file_content(self, file_content: list, file_id: str,
                            chunk_size: int=200, overlap_size: int=20):

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=overlap_size,
        #     length_function=len,
        # )
        text_splitter = SemanticChunker(OllamaEmbeddings(
            model='mxbai-embed-large:latest',
            base_url='http://192.168.1.2:8000'
        ),breakpoint_threshold_type="standard_deviation")

        file_content_texts = [
            rec.page_content
            for rec in file_content
        ]

        file_content_metadata = [
            rec.metadata
            for rec in file_content
        ]

        chunks = text_splitter.create_documents(
            file_content_texts,
            metadatas=file_content_metadata
        )

        return chunks

    # def process_file_content(self, file_content: list, chunk_size: int=100, overlap_size: int=20):
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size,
    #         chunk_overlap=overlap_size,
    #         length_function=len,
    #     )

    #     texts = [rec.page_content for rec in file_content]
    #     return text_splitter.create_documents(texts)

    import base64


    import requests
    from pathlib import Path




    # === Image Analysis ===
    def handle_image(self, file_path: str) -> dict:
        try:
            with open(file_path, "rb") as image_file:
                image_bytes = image_file.read()
            return self.handle_image_from_bytes(image_bytes, str(file_path))
        except Exception as e:
            return {
                "type": "image",
                "content": f"[Error during visual LLM analysis: {str(e)}]",
                "file_path": str(file_path)
            }

    # def handle_image_from_bytes(self, image_bytes: bytes, label: str) -> dict:
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

    #         response = requests.post("http://56.125.123.8:11434/api/generate", json=payload)
    #         result = response.json().get("response", "[No response returned]")

    #     except Exception as e:
    #         print(f"Error during visual LLM analysis: {str(e)}")
    #         result = f"[Error during visual LLM analysis: {str(e)}]"
            
    #         return [Document(
    #             page_content=result.strip(),
    #             metadata={
                    
    #                 "label": label,
    #                 "type": "image"
    #             }
    #         )]


    def handle_image(self, file_path: str) -> Optional[List[Document]]:
        try:
            # Read and convert image to base64
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Prepare API request
            prompt = (
                "Analyze this image and describe its contents in detail. "
                "Extract any visible text and describe key visual elements."
            )
            
            payload = {
                "model": "qwen2.5vl:7b",
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }

            # Make API request
            response = requests.post(
                f"http://56.125.123.8:11434/api/generate",
                json=payload,
                timeout=30
            )
            
            result = response.json().get("response", "")
            
            return [Document(
                page_content=result.strip(),
                metadata={"source": file_path}
            )]

        except Exception as e:
            # self.logger.error(f"Error processing image: {str(e)}")
            return None
        
    # # === PDF Handler with Inline Analysis ===
    # def handle_pdf_with_fallback(self,path: Path) -> dict:
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
        

    # def handle_image_from_bytes(self,image_bytes: bytes, label: str) -> dict:
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

