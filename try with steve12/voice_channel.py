from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
import os
import time
import uuid
import base64
import re
import httpx
import numpy as np

# Use the faster_whisper WhisperModel from the library.
from faster_whisper import WhisperModel
from PARLER_TTS import inference_parlertts
from chattertts import run_tts_pipeline

# --- Global fixed configuration for WhisperModel ---
# Use only supported parameters:
MODEL_SIZE = "tiny.en"  # this is now used as a model path identifier
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Global instance using only the parameters supported by ctranslate2.models.Whisper.
global_whisper_provider = WhisperModel(
    model_size_or_path=MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)
if not global_whisper_provider:
    raise RuntimeError("Failed to create WhisperModel instance.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice-to-Voice WebSocket Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define some common message types (you can expand these as needed)
from enum import Enum
class MessageType(str, Enum):
    INIT_SESSION = "init_session"
    START_LISTENING = "start_listening"
    STOP_LISTENING = "stop_listening"
    TRANSCRIPTION = "transcription"
    LLM_RESPONSE = "llm_response"
    TTS_AUDIO = "tts_audio"
    STATE_CHANGE = "state_change"
    STATUS = "status"
    ERROR = "error"

def clean_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

# --- A simple connection manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        # We'll use the global whisper provider for all sessions.
        self.voice_sessions: dict[str, WhisperModel] = {}
        self.session_configs: dict[str, dict] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.voice_sessions:
            # (If needed, clean-up per session; here we simply remove the reference.)
            del self.voice_sessions[client_id]
        if client_id in self.session_configs:
            del self.session_configs[client_id]
        logger.info(f"Client {client_id} disconnected and cleaned up")
        
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")

manager = ConnectionManager()

# --- A sample voice processing loop ---
# This loop now simulates transcription instead of calling wait_for_transcription.
async def voice_processing_loop(client_id: str):
    whisper_provider = global_whisper_provider
    client_id='wer'
    try:
        while client_id in manager.active_connections:
            # Replace the unavailable wait_for_transcription call
            # with your actual audio capture and transcription logic.
            # Here we simulate a waiting period and a dummy transcription.
            await asyncio.sleep(5)  # simulate waiting for audio and processing
            transcription_text = "This is a simulated transcription."
            if transcription_text:
                await manager.send_message(client_id, {
                    "type": MessageType.TRANSCRIPTION,
                    "text": clean_spaces(transcription_text),
                    "timestamp": time.time(),
                })
                # Process via LLM backend if needed (example below):
                llm_response = clean_spaces(transcription_text)
                await manager.send_message(client_id, {
                    "type": MessageType.LLM_RESPONSE,
                    "text": llm_response,
                    "timestamp": time.time(),
                })
                # Generate TTS from the LLM response:
                tts_start = time.time()
                audio_data = await run_tts_pipeline(llm_response, client_id)
                tts_end = time.time()
                if audio_data:
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                    await manager.send_message(client_id, {
                        "type": MessageType.TTS_AUDIO,
                        "audio": audio_base64,
                        "format": "pcm_int16",
                        "sample_rate": 16000,
                        "generation_time": tts_end - tts_start,
                        "timestamp": time.time(),
                    })
            else:
                await manager.send_message(client_id, {
                    "type": MessageType.STATUS,
                    "message": "No speech detected within timeout.",
                    "timestamp": time.time(),
                })
    except Exception as e:
        logger.error(f"Voice processing loop for client {client_id} encountered error: {e}")
        await manager.send_message(client_id, {
            "type": MessageType.ERROR,
            "message": str(e),
            "timestamp": time.time(),
        })

# --- WebSocket endpoint ---
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        # Send initial status
        await manager.send_message(client_id, {
            "type": MessageType.STATUS,
            "message": "Connected to voice-to-voice server",
            "client_id": client_id,
            "timestamp": time.time()
        })
        async for data in websocket.iter_text():
            message = json.loads(data)
            msg_type = message.get("type")
            if msg_type == MessageType.INIT_SESSION:
                # Do nothing regarding configuration—instead use our fixed global settings.
                manager.session_configs[client_id] = {
                    "project_id": "default_project",  # example fixed value
                    "tts_model": "chattertts"
                }
                # Associate the global whisper provider
                manager.voice_sessions[client_id] = global_whisper_provider
                await manager.send_message(client_id, {
                    "type": MessageType.STATUS,
                    "message": "Voice session initialized with fixed configuration.",
                    "config": {
                        "model_size_or_path": MODEL_SIZE,
                        "device": DEVICE,
                        "compute_type": COMPUTE_TYPE,
                    },
                    "timestamp": time.time()
                })
            elif msg_type == MessageType.START_LISTENING:
                if client_id in manager.voice_sessions:
                    # Start a background task for processing voice (e.g. transcription)
                    asyncio.create_task(voice_processing_loop(client_id))
                    await manager.send_message(client_id, {
                        "type": MessageType.STATUS,
                        "message": "Started voice processing.",
                        "timestamp": time.time()
                    })
                else:
                    await manager.send_message(client_id, {
                        "type": MessageType.ERROR,
                        "message": "Voice session not initialized.",
                        "timestamp": time.time()
                    })
            elif msg_type == MessageType.STOP_LISTENING:
                # In a more complete version, signal to stop processing
                await manager.send_message(client_id, {
                    "type": MessageType.STATUS,
                    "message": "Stopped voice processing.",
                    "timestamp": time.time()
                })
            else:
                await manager.send_message(client_id, {
                    "type": MessageType.ERROR,
                    "message": f"Unknown message type: {msg_type}",
                    "timestamp": time.time()
                })
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected via WebSocket.")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await manager.send_message(client_id, {
            "type": MessageType.ERROR,
            "message": str(e),
            "timestamp": time.time()
        })
    finally:
        manager.disconnect(client_id)

@app.get("/")
async def root():
    return {
        "message": "Voice-to-Voice WebSocket Server is running",
        "active_connections": len(manager.active_connections),
        "active_sessions": len(manager.voice_sessions),
    }

@app.get("/status")
async def status():
    return {
        "active_connections": len(manager.active_connections),
        "active_sessions": len(manager.voice_sessions),
        "clients": list(manager.active_connections.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_channel:app", host="localhost", port=7000, reload=True, log_level="info")