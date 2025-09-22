# from faster_whisper import WhisperModel
# import pyaudio
# import numpy as np
# import webrtcvad
# import logging
# import threading
# import queue
# import time
# import sounddevice as sd

# import torchaudio as ta
# import torch

# import  torchaudio as  ta

# from PARLER_TTS import inference_parlertts
# from chattertts import inference_chattertts


# class WhisperProvider:

#     def __init__(
#         self,
#         model_size: str = "medium",
#         device: str = "cpu",  # "cpu", "cuda", or "auto"
#         compute_type: str = "int8",  # "float16", "int8", "float32"
#         sample_rate: int = 16000,
#         chunk_duration: float = 0.03,  # 30ms chunks
#         vad_aggressiveness: int = 1,  # 0-3, higher = more aggressive
#         silence_timeout: float = 0.5,  # seconds of silence before processing
#         default_input_max_characters: int = 1000,
#         min_audio_length: float = 0.8,  # minimum audio length to process
#         confidence_threshold: float = 0.4  # minimum confidence to return result
#     ):
#         self.model_size = model_size
#         self.device = device
#         self.compute_type = compute_type
#         self.sample_rate = sample_rate
#         self.chunk_duration = chunk_duration
#         self.chunk_size = int(sample_rate * chunk_duration)
#         self.vad_aggressiveness = vad_aggressiveness
#         self.silence_timeout = silence_timeout
#         self.default_input_max_characters = default_input_max_characters
#         self.min_audio_length = min_audio_length
#         self.confidence_threshold = confidence_threshold
        
#         # Initialize components
#         self.whisper_model = None
#         self.vad = None
#         self.audio = None
#         self.stream = None
        
#         # Threading and queue for real-time processing
#         self.audio_queue = queue.Queue()
#         self.transcription_queue = queue.Queue()
#         self.is_recording = False
#         self.recording_thread = None
#         self.processing_thread = None
        
#         # Audio buffer for speech segments
#         self.speech_buffer = []
#         self.last_speech_time = 0
        
#         # Setup logging
#         logging.basicConfig(level=logging.INFO)
#         self.logger = logging.getLogger(__name__)

#     def initialize(self):
#         """Initialize Whisper model, VAD, and audio components"""
#         try:
#             # Load faster-whisper model
#             self.whisper_model = WhisperModel(
#                 self.model_size, 
#                 device=self.device, 
#                 compute_type=self.compute_type
#             )
#             self.logger.info(f"Loaded faster-whisper model: {self.model_size} on {self.device} with {self.compute_type}")
            
#             # Initialize VAD
#             self.vad = webrtcvad.Vad(self.vad_aggressiveness)
#             self.logger.info(f"Initialized VAD with aggressiveness: {self.vad_aggressiveness}")
            
#             # Initialize PyAudio
#             self.audio = pyaudio.PyAudio()
#             self.logger.info("Initialized PyAudio")
            
#             return True
            
#         except Exception as e:
#             self.logger.error(f"Error initializing Whisper provider: {str(e)}")
#             return False

#     def start_recording(self):
#         """Start real-time audio recording and processing"""
#         if not self.whisper_model or not self.vad or not self.audio:
#             self.logger.error("Components not initialized. Call initialize() first.")
#             return False
            
#         try:
#             # Open audio stream
#             self.stream = self.audio.open(
#                 format=pyaudio.paInt16,
#                 channels=1,
#                 rate=self.sample_rate,
#                 input=True,
#                 frames_per_buffer=self.chunk_size
#             )
            
#             self.is_recording = True
            
#             # Start recording thread
#             self.recording_thread = threading.Thread(target=self._record_audio)
#             self.recording_thread.daemon = True
#             self.recording_thread.start()
            
#             # Start processing thread
#             self.processing_thread = threading.Thread(target=self._process_audio)
#             self.processing_thread.daemon = True
#             self.processing_thread.start()
            
#             self.logger.info("Started real-time audio recording and processing")
#             return True
            
#         except Exception as e:
#             self.logger.error(f"Error starting recording: {str(e)}")
#             return False

#     def stop_recording(self):
#         """Stop real-time audio recording"""
#         self.is_recording = False
        
#         if self.stream:
#             self.stream.stop_stream()
#             self.stream.close()
            
#         if self.recording_thread:
#             self.recording_thread.join(timeout=1.0)
            
#         if self.processing_thread:
#             self.processing_thread.join(timeout=1.0)
            
#         self.logger.info("Stopped audio recording")

#     def _record_audio(self):
#         """Record audio in chunks and detect voice activity"""
#         while self.is_recording:
#             try:
#                 # Read audio chunk
#                 audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
#                 audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
#                 # Check if speech is detected
#                 is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
                
#                 if is_speech:
#                     self.speech_buffer.extend(audio_data)
#                     self.last_speech_time = time.time()
#                 else:
#                     # Check if we should process accumulated speech
#                     if (self.speech_buffer and 
#                         time.time() - self.last_speech_time > self.silence_timeout):
                        
#                         # Check minimum audio length before processing
#                         audio_duration = len(self.speech_buffer) / self.sample_rate
#                         if audio_duration >= self.min_audio_length:
#                             # Queue speech buffer for transcription
#                             speech_array = np.array(self.speech_buffer, dtype=np.float32)
#                             speech_array = speech_array / 32768.0  # Normalize to [-1, 1]
#                             self.audio_queue.put(speech_array.copy())
                        
#                         # Clear buffer
#                         self.speech_buffer = []
                        
#             except Exception as e:
#                 self.logger.error(f"Error in audio recording: {str(e)}")
#                 break

#     def _process_audio(self):
#         """Process queued audio chunks with Whisper"""
#         while self.is_recording:
#             try:
#                 # Get audio from queue (with timeout)
#                 try:
#                     audio_data = self.audio_queue.get(timeout=0.1)
#                 except queue.Empty:
#                     continue
                
#                 # Transcribe with faster-whisper
#                 segments, info = self.whisper_model.transcribe(
#                     audio_data,
#                     language="en",  # Auto-detect language
#                     task="transcribe",
#                     vad_filter=False,  # We're already using VAD
#                     word_timestamps=False
#                 )
                
#                 # Collect all segments
#                 transcription_parts = []
#                 for segment in segments:
#                     transcription_parts.append(segment.text)
                
#                 transcription = " ".join(transcription_parts).strip()
#                 detected_language = info.language
                
#                 # Only return transcription if it meets confidence threshold
#                 if transcription and info.language_probability >= self.confidence_threshold:
#                     # Process and queue transcription with language
#                     processed_text = self.process_text(transcription)
#                     transcription_result = {
#                         "text": processed_text,
#                         "language": detected_language,
#                         "language_probability": info.language_probability
#                     }
#                     self.transcription_queue.put(transcription_result)
#                     self.logger.info(f"Transcribed [{detected_language}:{info.language_probability:.2f}]: {processed_text}")
                
#                 self.audio_queue.task_done()
                
#             except Exception as e:
#                 self.logger.error(f"Error in audio processing: {str(e)}")

#     def get_transcription(self):
#         """Get the latest transcription with language (non-blocking)"""
#         try:
#             return self.transcription_queue.get_nowait()
#         except queue.Empty:
#             return None

#     def get_transcription_text_only(self):
#         """Get only the transcription text for LLM (non-blocking)"""
#         result = self.get_transcription()
#         return result["text"] if result else None

#     def wait_for_transcription(self, timeout=10):
#         """Wait for next transcription (blocking with timeout) - Perfect for LLM integration"""
#         try:
#             result = self.transcription_queue.get(timeout=timeout)
#             return result["text"]
#         except queue.Empty:
#             return None

#     def get_all_transcriptions(self):
#         """Get all pending transcriptions with languages"""
#         transcriptions = []
#         while True:
#             try:
#                 transcription = self.transcription_queue.get_nowait()
#                 transcriptions.append(transcription)
#             except queue.Empty:
#                 break
#         return transcriptions

#     def process_text(self, text: str):
#         """Process transcribed text - clean for LLM consumption"""
#         # Remove extra whitespace and limit length
#         text = text.strip()
#         text = ' '.join(text.split())  # Remove multiple spaces
#         return text[:self.default_input_max_characters]

#     def transcribe_file(self, audio_file_path: str):
#         """Transcribe an audio file with language detection (utility method)"""
#         if not self.whisper_model:
#             self.logger.error("Whisper model not initialized")
#             return None
            
#         try:
#             segments, info = self.whisper_model.transcribe(
#                 audio_file_path,
#                 language=None,
#                 task="transcribe"
#             )
            
#             # Collect all segments
#             transcription_parts = []
#             for segment in segments:
#                 transcription_parts.append(segment.text)
            
#             full_transcription = " ".join(transcription_parts)
            
#             return {
#                 "text": self.process_text(full_transcription),
#                 "language": info.language,
#                 "language_probability": info.language_probability
#             }
            
#         except Exception as e:
#             self.logger.error(f"Error transcribing file: {str(e)}")
#             return None

#     def cleanup(self):
#         """Clean up resources"""
#         self.stop_recording()
        
#         if self.audio:
#             self.audio.terminate()
            
#         self.logger.info("Cleaned up Whisper provider resources")

# import httpx
# import asyncio
# from datetime import datetime
# import time
# # Example usage for LLM integration
# if __name__ == "__main__":
#     # tts_model = ChatterboxTTS.from_pretrained(device="cpu")   

#     # Create and initialize the provider
#     whisper_provider = WhisperProvider(
#         model_size="tiny",
#         device="cpu",
#         compute_type="int8",
#         silence_timeout=1.0,  # Wait 2 seconds after speech ends
#         min_audio_length=.8,  # Ignore very short audio
#         confidence_threshold=.5  # Filter low-confidence results
#     )
    
#     if not whisper_provider.initialize():
#         print("Failed to initialize Whisper provider")
#         exit(1)
    
#     print("Starting real-time transcription for LLM...")
#     print("Speak into your microphone. The system will wait for silence before processing.")
#     print("Press Ctrl+C to stop.")
    
#     # Start recording
#     if not whisper_provider.start_recording():
#         print("Failed to start recording")
#         exit(1)
    


#     project_id = "wer"  # Replace with your actual project_id

#     try:
#         while True:
#             print("Waiting for speech...")
#             transcription_text = whisper_provider.wait_for_transcription(timeout=30)
#             start_time = time.time()
#             if transcription_text:

#                 print(f"[Transcription] {transcription_text}")

#                 async def call_rag():
#                     async with httpx.AsyncClient(timeout=60.0) as client:
#                         response = await client.post(
#                             f"http://localhost:5000/api/v1/nlp/index/answer/{project_id}",
#                             json={"text": transcription_text, "limit": 5}
#                         )

#                         if response.status_code == 200:
                            
#                             # print(f"\n[LLM Answer] {data['answer']}\n")

                            
#                             answer = response.json()["answer"]
#                             print(answer)

#                             tts_start = time.time()
#                             wav =await inference_parlertts(str(answer),'wer')
#                             tts_end=time.time()
#                             print(f"TTS parler_tts: {tts_end - tts_start:.2f} seconds")

#                             tts_start = time.time()
#                             wav =  await inference_chattertts(str(answer),'wer')
#                             tts_end=time.time()
#                             print(f"TTS chater_tts: {tts_end - tts_start:.2f} seconds")

#                         else:
#                             print(f"[RAG Error] {response.status_code}: {response.text}")

#                 asyncio.run(call_rag())
#             else:
#                 print("No speech detected in 30 seconds")

#     except KeyboardInterrupt:
#         print("\nStopping transcription...")

#     finally:
#         whisper_provider.cleanup()
#         print("Transcription stopped.")

from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import webrtcvad
import logging
import threading
import queue
import time
import sounddevice as sd
from enum import Enum
import asyncio

import torchaudio as ta
import torch

from chattertts import run_tts_pipeline


class AudioState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class WhisperProvider:

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",  # "cpu", "cuda", or "auto"
        compute_type: str = "int8",  # "float16", "int8", "float32"
        sample_rate: int = 16000,
        chunk_duration: float = 0.03,  # 30ms chunks
        vad_aggressiveness: int = 1,  # 0-3, higher = more aggressive
        silence_timeout: float = 0.5,  # seconds of silence before processing
        default_input_max_characters: int = 1000,
        min_audio_length: float = 0.8,  # minimum audio length to process
        confidence_threshold: float = 0.4,  # minimum confidence to return result
        state_transition_delay: float = 0.15  # delay between state transitions
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.vad_aggressiveness = vad_aggressiveness
        self.silence_timeout = silence_timeout
        self.default_input_max_characters = default_input_max_characters
        self.min_audio_length = min_audio_length
        self.confidence_threshold = confidence_threshold
        self.state_transition_delay = state_transition_delay
        
        # Initialize components
        self.whisper_model = None
        self.vad = None
        self.audio = None
        self.stream = None
        
        # Threading and queue for real-time processing
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.is_recording = False
        self.recording_thread = None
        self.processing_thread = None
        
        # Audio buffer for speech segments
        self.speech_buffer = []
        self.last_speech_time = 0
        
        # State management
        self.current_state = AudioState.IDLE
        self.state_lock = threading.RLock()  # Reentrant lock for nested calls
        self.state_change_event = threading.Event()  # For signaling state changes
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def set_state(self, new_state: AudioState):
        """Thread-safe state setter with logging and cleanup"""
        with self.state_lock:
            if self.current_state == new_state:
                return
                
            old_state = self.current_state
            self.logger.info(f"State transition: {old_state.value} → {new_state.value}")
            
            # Cleanup actions when leaving certain states
            if old_state == AudioState.LISTENING and new_state in [AudioState.PROCESSING, AudioState.SPEAKING]:
                self._clear_audio_buffers()
                
            # Set new state
            self.current_state = new_state
            self.state_change_event.set()  # Signal waiting threads
            self.state_change_event.clear()
            
            # Brief delay to allow audio hardware to settle
            if self.state_transition_delay > 0:
                time.sleep(self.state_transition_delay)

    def get_state(self) -> AudioState:
        """Thread-safe state getter"""
        with self.state_lock:
            return self.current_state

    def _clear_audio_buffers(self):
        """Clear all audio buffers - call this when switching away from LISTENING"""
        self.speech_buffer = []
        self.last_speech_time = 0
        
        # Clear the audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        self.logger.debug("Audio buffers cleared")

    def _should_process_audio(self) -> bool:
        """Check if we should process incoming audio based on current state"""
        current_state = self.get_state()
        return current_state in [AudioState.LISTENING]

    def _should_transcribe(self) -> bool:
        """Check if we should transcribe queued audio"""
        current_state = self.get_state()
        return current_state in [AudioState.LISTENING, AudioState.PROCESSING]

    def initialize(self):
        """Initialize Whisper model, VAD, and audio components"""
        try:
            # Load faster-whisper model
            self.whisper_model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type
            )
            self.logger.info(f"Loaded faster-whisper model: {self.model_size} on {self.device} with {self.compute_type}")
            
            # Initialize VAD
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)
            self.logger.info(f"Initialized VAD with aggressiveness: {self.vad_aggressiveness}")
            
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            self.logger.info("Initialized PyAudio")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Whisper provider: {str(e)}")
            return False

    def start_recording(self):
        """Start real-time audio recording and processing"""
        if not self.whisper_model or not self.vad or not self.audio:
            self.logger.error("Components not initialized. Call initialize() first.")
            return False
            
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Set initial state to listening
            self.set_state(AudioState.LISTENING)
            
            self.logger.info("Started real-time audio recording and processing")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting recording: {str(e)}")
            return False

    def stop_recording(self):
        """Stop real-time audio recording"""
        self.set_state(AudioState.IDLE)
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
            
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
        self.logger.info("Stopped audio recording")

    def _record_audio(self):
        """Record audio in chunks and detect voice activity"""
        while self.is_recording:
            try:
                # Read audio chunk
                audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Check if we should process audio based on current state
                if not self._should_process_audio():
                    continue  # Skip processing but keep reading to prevent buffer overflow
                
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Check if speech is detected
                is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
                
                if is_speech:
                    self.speech_buffer.extend(audio_data)
                    self.last_speech_time = time.time()
                else:
                    # Check if we should process accumulated speech
                    if (self.speech_buffer and 
                        time.time() - self.last_speech_time > self.silence_timeout):
                        
                        # Double-check state before processing (it might have changed)
                        if self._should_process_audio():
                            # Check minimum audio length before processing
                            audio_duration = len(self.speech_buffer) / self.sample_rate
                            if audio_duration >= self.min_audio_length:
                                # Queue speech buffer for transcription
                                speech_array = np.array(self.speech_buffer, dtype=np.float32)
                                speech_array = speech_array / 32768.0  # Normalize to [-1, 1]
                                self.audio_queue.put(speech_array.copy())
                        
                        # Clear buffer regardless of state
                        self.speech_buffer = []
                        
            except Exception as e:
                self.logger.error(f"Error in audio recording: {str(e)}")
                break

    def _process_audio(self):
        """Process queued audio chunks with Whisper"""
        while self.is_recording:
            try:
                # Get audio from queue (with timeout)
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Check if we should transcribe based on current state
                if not self._should_transcribe():
                    self.audio_queue.task_done()
                    continue
                
                # Transcribe with faster-whisper
                segments, info = self.whisper_model.transcribe(
                    audio_data,
                    language="en",  # Auto-detect language
                    task="transcribe",
                    vad_filter=False,  # We're already using VAD
                    word_timestamps=False
                )
                
                # Collect all segments
                transcription_parts = []
                for segment in segments:
                    transcription_parts.append(segment.text)
                
                transcription = " ".join(transcription_parts).strip()
                detected_language = info.language
                
                # Only return transcription if it meets confidence threshold and we're still in correct state
                if (transcription and 
                    info.language_probability >= self.confidence_threshold and 
                    self._should_transcribe()):
                    
                    # Process and queue transcription with language
                    processed_text = self.process_text(transcription)
                    transcription_result = {
                        "text": processed_text,
                        "language": detected_language,
                        "language_probability": info.language_probability
                    }
                    self.transcription_queue.put(transcription_result)
                    self.logger.info(f"Transcribed [{detected_language}:{info.language_probability:.2f}]: {processed_text}")
                
                self.audio_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in audio processing: {str(e)}")

    def get_transcription(self):
        """Get the latest transcription with language (non-blocking)"""
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return None

    def get_transcription_text_only(self):
        """Get only the transcription text for LLM (non-blocking)"""
        result = self.get_transcription()
        return result["text"] if result else None

    def wait_for_transcription(self, timeout=10):
        """Wait for next transcription (blocking with timeout) - Perfect for LLM integration"""
        try:
            result = self.transcription_queue.get(timeout=timeout)
            return result["text"]
        except queue.Empty:
            return None

    def get_all_transcriptions(self):
        """Get all pending transcriptions with languages"""
        transcriptions = []
        while True:
            try:
                transcription = self.transcription_queue.get_nowait()
                transcriptions.append(transcription)
            except queue.Empty:
                break
        return transcriptions

    def process_text(self, text: str):
        """Process transcribed text - clean for LLM consumption"""
        # Remove extra whitespace and limit length
        text = text.strip()
        text = ' '.join(text.split())  # Remove multiple spaces
        return text[:self.default_input_max_characters]

    def transcribe_file(self, audio_file_path: str):
        """Transcribe an audio file with language detection (utility method)"""
        if not self.whisper_model:
            self.logger.error("Whisper model not initialized")
            return None
            
        try:
            segments, info = self.whisper_model.transcribe(
                audio_file_path,
                language=None,
                task="transcribe"
            )
            
            # Collect all segments
            transcription_parts = []
            for segment in segments:
                transcription_parts.append(segment.text)
            
            full_transcription = " ".join(transcription_parts)
            
            return {
                "text": self.process_text(full_transcription),
                "language": info.language,
                "language_probability": info.language_probability
            }
            
        except Exception as e:
            self.logger.error(f"Error transcribing file: {str(e)}")
            return None

    def cleanup(self):
        """Clean up resources"""
        self.stop_recording()
        
        if self.audio:
            self.audio.terminate()
            
        self.logger.info("Cleaned up Whisper provider resources")


# Enhanced main loop with proper state management
import httpx
import asyncio
import time

async def main():
    # Create and initialize the provider
    whisper_provider = WhisperProvider(
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        silence_timeout=1.0,
        min_audio_length=0.8,
        confidence_threshold=0.5,
        state_transition_delay=0.2  # 200ms delay between state changes
    )
    
    if not whisper_provider.initialize():
        print("Failed to initialize Whisper provider")
        exit(1)
    
    print("Starting real-time transcription with state management...")
    print("States: LISTENING → PROCESSING → SPEAKING → LISTENING")
    print("Press Ctrl+C to stop.")
    
    # Start recording
    if not whisper_provider.start_recording():
        print("Failed to start recording")
        exit(1)

    project_id = "wer"

    try:
        while True:
            print(f"\n[State: {whisper_provider.get_state().value.upper()}] Waiting for speech...")
            
            # Wait for transcription (this happens in LISTENING state)
            transcription_text = whisper_provider.wait_for_transcription(timeout=30)
            
            if transcription_text:
                print(f"[Transcription] {transcription_text}")
                
                # Switch to PROCESSING state - STT stops listening
                whisper_provider.set_state(AudioState.PROCESSING)
                
                start_time = time.time()
                
                async def call_rag():
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        try:
                            tts_start = time.time()

                            response = await client.post(
                                f"http://localhost:5000/api/v1/nlp/index/answer/{project_id}",
                                json={"text": transcription_text, "limit": 5}
                            )

                            if response.status_code == 200:
                                answer = response.json()["answer"]
                                print(f"[LLM Answer] {answer}")

                                # Switch to SPEAKING state - TTS will play
                                whisper_provider.set_state(AudioState.SPEAKING)
                                import re

                                def clean_spaces(text: str) -> str:
                                    # Replace multiple spaces/newlines/tabs with a single space
                                    return re.sub(r'\s+', ' ', text).strip()

                                answer = clean_spaces(answer)
                                tts_end = time.time()
                                print(f"RAG call: {tts_end - tts_start:.2f} seconds")

                                tts_start = time.time()
    # Run the pipeline
                                await run_tts_pipeline(
                                    text=answer,
                                    audio_prompt_path=None,  # Use default voice
                                    temperature=0.8
                                )
                                tts_end = time.time()
                                print(f"TTS chater_tts: {tts_end - tts_start:.2f} seconds")
                                
                                # TTS finished, return to LISTENING state
                                whisper_provider.set_state(AudioState.LISTENING)
                                print("[State: LISTENING] Ready for next input...")

                            else:
                                print(f"[RAG Error] {response.status_code}: {response.text}")
                                # Return to listening even on error
                                whisper_provider.set_state(AudioState.LISTENING)
                                
                        except Exception as e:
                            print(f"[Error] {str(e)}")
                            # Return to listening on any error
                            whisper_provider.set_state(AudioState.LISTENING)

                await call_rag()
                
            else:
                print("No speech detected in 30 seconds")
                # Ensure we're still in listening state
                whisper_provider.set_state(AudioState.LISTENING)

    except KeyboardInterrupt:
        print("\nStopping transcription...")

    finally:
        whisper_provider.cleanup()
        print("Transcription stopped.")


if __name__ == "__main__":
    asyncio.run(main())