

import threading
import queue
import re
import time
import numpy as np
import torch
import sounddevice as sd

# We import the UNMODIFIED ChatterboxTTS class
from chatterbox.tts import ChatterboxTTS

# --- Step 1: A Naive Text Chunker ---
def chunk_text_naively(text: str) -> list[str]:
    """
    A simple function to chunk text by sentence-ending punctuation.
    This is a "naive" approach as it doesn't handle complex cases like abbreviations.
    """
    # Use regex to split text by periods, question marks, or exclamation marks, keeping the delimiters
    sentences = re.split(r'([.?!])', text)
    
    # The result of the split is [sentence1, '.', sentence2, '?', ...]. We need to join them back.
    chunks = []
    if len(sentences) > 1:
        for i in range(0, len(sentences) - 1, 2):
            chunk = sentences[i] + sentences[i+1]
            if chunk.strip(): # Avoid adding empty strings
                chunks.append(chunk.strip())
    else:
        # Handle case where there's no punctuation
        if text.strip():
            chunks.append(text.strip())
            
    # If the last part of the text had no punctuation, add it.
    if len(sentences) % 2 != 0 and sentences[-1].strip():
        chunks.append(sentences[-1].strip())
        
    return chunks

# --- Step 2: An Audio Player Class to handle the Consumer side ---
class AudioPlayer:
    """
    Manages the audio playback in a separate thread using a callback.
    This is the "Consumer" part of our pipeline.
    """
    def __init__(self, sample_rate: int, audio_queue: queue.Queue):
        self.sample_rate = sample_rate
        self.audio_queue = audio_queue
        self.stream = None
        self.is_playing = False
        self._producer_finished = False
        self._playback_buffer = np.array([], dtype=np.float32)

    def _audio_callback(self, outdata: np.ndarray, frames: int, time, status):
        """This function is called by the sounddevice library to get more audio data."""
        if status:
            print(f"Playback status warning: {status}")

        chunk_size = len(outdata)
        
        # If we have enough data in our buffer, send it
        if len(self._playback_buffer) >= chunk_size:
            outdata[:] = self._playback_buffer[:chunk_size].reshape(-1, 1)
            self._playback_buffer = self._playback_buffer[chunk_size:]
            return

        # If not, try to get more data from the generation queue
        while len(self._playback_buffer) < chunk_size:
            try:
                # Get the next audio chunk. The timeout prevents blocking forever.
                next_chunk = self.audio_queue.get_nowait()
                if next_chunk is None:  # Sentinel value indicates the end
                    self._producer_finished = True
                    break
                self._playback_buffer = np.concatenate((self._playback_buffer, next_chunk))
            except queue.Empty:
                # Queue is empty, producer is likely busy. Fill with silence.
                outdata[:] = np.zeros((chunk_size, 1), dtype=np.float32)
                return

        # If we broke the loop because the producer finished and buffer is empty
        if self._producer_finished and len(self._playback_buffer) == 0:
            outdata.fill(0)
            raise sd.CallbackStop # Signal that playback is complete

        # We have some data, but maybe not enough. Pad with silence.
        data_to_play = self._playback_buffer[:chunk_size]
        self._playback_buffer = self._playback_buffer[chunk_size:]
        outdata[:len(data_to_play)] = data_to_play.reshape(-1, 1)
        outdata[len(data_to_play):].fill(0)

    def start(self):
        """Starts the audio playback stream."""
        if self.is_playing:
            return
        print("Starting audio playback stream...")
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            dtype='float32'
        )
        self.stream.start()
        self.is_playing = True

    def stop(self):
        """Stops the audio playback stream."""
        if not self.is_playing or self.stream is None:
            return
        print("Stopping audio playback stream...")
        self.stream.stop()
        self.stream.close()
        self.is_playing = False

    def is_finished(self) -> bool:
        """Checks if the producer has finished and the playback buffer is empty."""
        return self._producer_finished and len(self._playback_buffer) == 0


# ==============================================================================
# REPLACE the old generation_thread_func in your script with this new version.
# ==============================================================================

import time # Make sure 'time' is imported at the top of your script

def generation_thread_func(tts_instance: ChatterboxTTS, text: str, audio_queue: queue.Queue, generation_kwargs: dict):
    """
    This function runs in a background thread to generate audio chunks.
    It now includes detailed timing and content logging for each chunk.
    """
    try:
        if "audio_prompt_path" in generation_kwargs and generation_kwargs["audio_prompt_path"]:
            print("Preparing voice conditionals from audio prompt...")
            tts_instance.prepare_conditionals(generation_kwargs["audio_prompt_path"])
        
        text_chunks = chunk_text_naively(text)
        print(f"Text split into {len(text_chunks)} chunks for generation.")

        # --- NEW: Variables to track overall performance ---
        total_generation_time = 0
        total_audio_duration = 0

        # Generate each chunk and put it in the queue
        for i, chunk in enumerate(text_chunks):
            # --- NEW: Print chunk content and start timer ---
            print(f"\n--- Generating chunk {i+1}/{len(text_chunks)} ---")
            print(f"  - Content: '{chunk}'")
            
            start_time = time.time()
            
            # This is the actual call to the TTS model
            audio_tensor = tts_instance.generate(text=chunk, **generation_kwargs)
            
            # --- NEW: Stop timer and calculate metrics ---
            end_time = time.time()
            
            generation_time = end_time - start_time
            total_generation_time += generation_time

            audio_numpy = audio_tensor.squeeze().cpu().numpy()
            audio_duration = len(audio_numpy) / tts_instance.sr
            total_audio_duration += audio_duration
            
            # Avoid division by zero if an empty audio chunk is somehow generated
            rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

            # --- NEW: Print detailed metrics for the chunk ---
            print(f"  - Time to generate: {generation_time:.2f} seconds")
            print(f"  - Audio duration of chunk: {audio_duration:.2f} seconds")
            print(f"  - Real-Time Factor (RTF): {rtf:.2f}")

            # Put the generated audio into the queue for playback
            audio_queue.put(audio_numpy)
            
    finally:
        # --- NEW: Print a final summary of the entire generation process ---
        print("\n" + "="*50)
        print("                Generation Summary")
        print("="*50)
        print(f"Total text chunks processed: {len(text_chunks)}")
        print(f"Total audio duration generated: {total_audio_duration:.2f} seconds")
        print(f"Total time spent on generation: {total_generation_time:.2f} seconds")
        
        overall_rtf = total_generation_time / total_audio_duration if total_audio_duration > 0 else float('inf')
        print(f"Overall Real-Time Factor (RTF): {overall_rtf:.2f}")
        print("="*50)

        # Signal the end of generation by putting None in the queue
        print("\nGeneration finished. Signaling playback to end.")
        audio_queue.put(None)

# ...existing code...

async def run_tts_pipeline(text: str, audio_prompt_path: str = None, temperature: float = 0.8):
    """
    Runs the complete TTS pipeline with streaming audio generation and playback.
    
    Args:
        text (str): The text to convert to speech
        audio_prompt_path (str, optional): Path to voice prompt WAV file. Defaults to None.
        temperature (float, optional): Generation temperature. Defaults to 0.8.
    """
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize TTS
    print("Loading ChatterboxTTS model... (This may take a moment)")
    tts = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded successfully.")

    # Setup audio queue and player
    audio_queue = queue.Queue()
    audio_player = AudioPlayer(sample_rate=tts.sr, audio_queue=audio_queue)

    # Setup generation arguments
    generation_args = {
        "audio_prompt_path": audio_prompt_path,
        "temperature": temperature,
    }

    # Create and start the producer thread
    producer_thread = threading.Thread(
        target=generation_thread_func,
        args=(tts, text, audio_queue, generation_args)
    )

    try:
        # Start audio playback and generation
        audio_player.start()
        producer_thread.start()

        # Keep the main thread alive
        while not audio_player.is_finished() or producer_thread.is_alive():
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Clean shutdown
        if producer_thread.is_alive():
            print("Waiting for generation thread to finish...")
            producer_thread.join()
        audio_player.stop()
        print("Playback finished.")
# Add this line to explicitly export the function
__all__ = ['run_tts_pipeline']
# Modified main block to use the new function
if __name__ == "__main__":
    import asyncio
    
    # Example text
    LONG_TEXT_PROMPT = (
        "Hello and welcome. This is a demonstration of asynchronous, real-time "
        "text-to-speech generation. The system works by breaking the text into "
        "smaller sentences. Each sentence is synthesized into audio independently. "
        "While one piece of audio is playing, the next is being generated in the "
        "background. This allows you to hear the output almost instantly, without "
        "waiting for the entire paragraph to be processed. This concludes the demonstration."
    )

    # Run the pipeline
    asyncio.run(run_tts_pipeline(
        text=LONG_TEXT_PROMPT,
        audio_prompt_path=None,  # Use default voice
        temperature=0.8
    ))