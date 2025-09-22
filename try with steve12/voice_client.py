import asyncio
import json
import uuid
import websockets
import base64
import numpy as np
import sounddevice as sd
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the message types that match those in voice_channel
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

class VoiceWebSocketClient:
    """WebSocket client for voice-to-voice communication"""
    def __init__(self, server_url: str = "ws://localhost:7000"):
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())
        self.websocket = None
        self.is_connected = False

    async def connect(self):
        try:
            self.websocket = await websockets.connect(f"{self.server_url}/ws/{self.client_id}")
            self.is_connected = True
            logger.info(f"Connected to server with client_id: {self.client_id}")
            asyncio.create_task(self.handle_messages())
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    async def disconnect(self):
        self.is_connected = False
        if self.websocket:
            await self.websocket.close()

    async def send_message(self, message: dict):
        if self.websocket and self.is_connected:
            await self.websocket.send(json.dumps(message))

    async def handle_messages(self):
        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get("type")
                if message_type == MessageType.TRANSCRIPTION:
                    print(f"\n📝 Transcription: {data['text']}")
                elif message_type == MessageType.LLM_RESPONSE:
                    print(f"🤖 LLM Response: {data['text']}")
                elif message_type == MessageType.TTS_AUDIO:
                    print(f"🔊 Playing TTS audio (generated in {data.get('generation_time', 0):.2f}s)")
                    await self.play_audio(data)
                elif message_type == MessageType.STATE_CHANGE:
                    print(f"🔄 State: {data.get('state', '').upper()}")
                elif message_type == MessageType.STATUS:
                    print(f"ℹ️  Status: {data.get('message', '')}")
                elif message_type == MessageType.ERROR:
                    err_msg = data.get('message', '')
                    # Customize error handling for missing transcription method.
                    if "wait_for_transcription" in err_msg:
                        print("❌ Error: Server transcription function is not available. "
                              "Please verify that the voice_channel server is updated to call the correct transcription method.")
                    else:
                        print(f"❌ Error: {err_msg}")
                else:
                    print(f"📨 Unknown message type: {message_type}")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed by server")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def play_audio(self, audio_data: dict):
        try:
            audio_bytes = base64.b64decode(audio_data['audio'])
            if audio_data.get('format') == 'pcm_int16':
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32767.0
            else:
                audio_float = np.frombuffer(audio_bytes, dtype=np.float32)
            sample_rate = audio_data.get('sample_rate', 16000)
            sd.play(audio_float, sample_rate)
            sd.wait()
        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    async def init_session(self):
        # Send a simple init session message without additional configuration.
        await self.send_message({"type": MessageType.INIT_SESSION})

    async def start_listening(self):
        await self.send_message({"type": MessageType.START_LISTENING})

    async def stop_listening(self):
        await self.send_message({"type": MessageType.STOP_LISTENING})

async def main():
    client = VoiceWebSocketClient("ws://localhost:7000")
    try:
        await client.connect()
        print("🔧 Initializing voice session...")
        await client.init_session()
        await asyncio.sleep(2)
        print("🎤 Starting voice processing...")
        print("Speak into your microphone. The system will process your speech and respond.")
        print("Press Ctrl+C to stop.")
        await client.start_listening()
        while client.is_connected:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping client...")
        await client.stop_listening()
    except Exception as e:
        logger.error(f"Client error: {e}")
    finally:
        await client.disconnect()
        print("👋 Disconnected from server")

if __name__ == "__main__":
    asyncio.run(main())