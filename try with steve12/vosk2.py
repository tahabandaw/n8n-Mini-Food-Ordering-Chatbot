from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import webrtcvad
import time
import httpx
import asyncio
import sounddevice as sd
from chatterbox.tts import ChatterboxTTS

# --- Settings ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.03
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
VAD_MODE = 1
SILENCE_TIMEOUT = 0.5
MIN_AUDIO_LENGTH = 0.8
CONFIDENCE_THRESHOLD = 0.4
PROJECT_ID = "5351"

# Load models
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
tts_model = ChatterboxTTS.from_pretrained(device="cuda")
vad = webrtcvad.Vad(VAD_MODE)
pa = pyaudio.PyAudio()

# Open mic
stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)

print("Speak into your microphone. Ctrl+C to exit.")

speech_buffer = []
last_speech_time = 0

async def call_rag_and_tts(text):
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"http://localhost:5000/api/v1/nlp/index/answer/{PROJECT_ID}",
            json={"text": text, "limit": 5}
        )
        if r.status_code == 200:
            answer = r.json()["answer"]
            print(f"[RAG Answer] {answer}")
            wav = tts_model.generate(str(answer))
            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            # Ensure NumPy float32
            wav = wav.astype("float32")

            sd.play(wav, samplerate=22050, blocking=True)
            sd.wait()
        else:
            print(f"Error from RAG: {r.status_code} - {r.text}")

try:
    while True:
        chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        is_speech = vad.is_speech(chunk, SAMPLE_RATE)

        if is_speech:
            speech_buffer.extend(np.frombuffer(chunk, dtype=np.int16))
            last_speech_time = time.time()
        else:
            if speech_buffer and time.time() - last_speech_time > SILENCE_TIMEOUT:
                audio_len = len(speech_buffer) / SAMPLE_RATE
                if audio_len >= MIN_AUDIO_LENGTH:
                    audio_array = np.array(speech_buffer, dtype=np.float32) / 32768.0
                    segments, info = whisper_model.transcribe(audio_array)

                    text = " ".join(seg.text for seg in segments).strip()
                    if text and info.language_probability >= CONFIDENCE_THRESHOLD:
                        print(f"[You said] {text}")
                        asyncio.run(call_rag_and_tts(text))
                speech_buffer = []

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
