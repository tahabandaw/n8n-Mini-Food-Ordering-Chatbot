from ..LLMInterface import LLMInterface
from ..LLMEnums import OllamaEnums
from ollama import Client
import logging
# import httpx

class OllamaProvider(LLMInterface):
    def __init__(
        self,
        host: str = "http://0.0.0.0:8000",
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1
    ):
        self.host = host.strip() if host else None
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        # Configure HTTP client
        # self.http_client = httpx.Client()

        # Initialize Ollama client
        self.client = Client(
            host=self.host
        )

        self.enums = OllamaEnums
        self.logger = logging.getLogger(__name__)
        
        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        
        if not self.client:
            self.logger.error("Ollama client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Ollama was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        # Convert chat history to Ollama format
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in chat_history
        ]
        messages.append(self.construct_prompt(prompt=prompt, role=OllamaEnums.USER.value))

        try:
            response = self.client.chat(
                model=self.generation_model_id,
                messages=messages,
                options={
                    "num_predict": max_output_tokens,
                    "temperature": temperature
                }
            )
            
            if not response or "message" not in response:
                self.logger.error("Error while generating text with Ollama")
                return None

            return response["message"]["content"]

        except Exception as e:
            self.logger.error(f"Error while generating text with Ollama: {str(e)}")
            return None

    def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("Ollama client was not set")
            return None

        if not self.embedding_model_id:
            
            self.logger.error("Embedding model for Ollama was not set")
            return None

        try:
            response = self.client.embeddings(
                model=self.embedding_model_id,
                prompt=text
            )

            if not response or "embedding" not in response:
                self.logger.error("Error while embedding text with Ollama")

                return None

            return response["embedding"]

        except Exception as e:
            self.logger.error(f"Error while embedding text with Ollama: {str(e)}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }