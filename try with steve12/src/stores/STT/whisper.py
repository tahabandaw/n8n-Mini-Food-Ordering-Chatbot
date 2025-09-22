from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import webrtcvad
import logging
import threading
import queue
import time

class WhisperProvider:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",  # "cpu", "cuda", or "auto"
        compute_type: str = "int8",  # "float16", "int8", "float32"
        sample_rate: int = 16000,
        chunk_duration: float = 0.03,  # 30ms chunks
        vad_aggressiveness: int = 2,  # 0-3, higher = more aggressive
        silence_timeout: float = 1.0,  # seconds of silence before processing
        default_input_max_characters: int = 1000
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
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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
            
            self.logger.info("Started real-time audio recording and processing")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting recording: {str(e)}")
            return False

    def stop_recording(self):
        """Stop real-time audio recording"""
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
                        
                        # Queue speech buffer for transcription
                        speech_array = np.array(self.speech_buffer, dtype=np.float32)
                        speech_array = speech_array / 32768.0  # Normalize to [-1, 1]
                        self.audio_queue.put(speech_array.copy())
                        
                        # Clear buffer
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
                
                # Transcribe with faster-whisper
                if len(audio_data) > self.sample_rate * 0.1:  # Minimum 100ms of audio
                    segments, info = self.whisper_model.transcribe(
                        audio_data,
                        language=None,  # Auto-detect language
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
                    
                    if transcription:
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
        """Process transcribed text"""
        return text[:self.default_input_max_characters].strip()

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


# Example usage
if __name__ == "__main__":
    # Create and initialize the provider with faster-whisper optimizations
    whisper_provider = WhisperProvider(
        model_size="base",
        device="cpu",  # Will use CUDA if available, otherwise CPU
        compute_type="int8"  # More efficient than float32
    )
    
    if not whisper_provider.initialize():
        print("Failed to initialize Whisper provider")
        exit(1)
    
    print("Starting real-time transcription with faster-whisper...")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    
    # Start recording
    if not whisper_provider.start_recording():
        print("Failed to start recording")
        exit(1)
    
    try:
        # Main loop to get transcriptions
        while True:
            result = whisper_provider.get_transcription()
            if result:
                print(f"[{result['language']}:{result['language_probability']:.2f}] {result['text']}")
            time.sleep(0.1)  # Small delay to prevent busy waiting
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
        
    finally:
        # Clean up
        whisper_provider.cleanup()
        print("Transcription stopped.")