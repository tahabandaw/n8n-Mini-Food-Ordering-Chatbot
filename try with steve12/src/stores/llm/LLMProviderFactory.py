
from .LLMEnums import LLMEnums
from .providers.OllamaProvider import OllamaProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):

        if provider == LLMEnums.OLLAMA.value:
            return OllamaProvider(
                host = self.config.OLLAMA_HOST,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
            )

        return None
