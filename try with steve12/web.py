import asyncio
import websockets
import json

# This set will store all connected WebSocket clients
connected_clients = set()

async def handler(websocket, path):
    """
    Handle WebSocket connections, register clients, and broadcast messages.
    """
    # Add the new client to our set of connected clients
    connected_clients.add(websocket)
    print(f"New client connected: {websocket.remote_address}")
    
    try:
        # Keep the connection open and listen for messages
        async for message in websocket:
            # When a message is received, broadcast it to all other clients
            # This allows the main app to send transcriptions to all clients
            websockets.broadcast(connected_clients, message)
            
    except websockets.exceptions.ConnectionClosedError:
        print(f"Client disconnected: {websocket.remote_address}")
    finally:
        # Remove the client from the set when they disconnect
        connected_clients.remove(websocket)

async def main():
    """
    Start the WebSocket server.
    """
    # Set the host and port for the server
    host = "localhost"
    port = 8765
    
    print(f"Starting WebSocket server on ws://{host}:{port}...")
    
    # Start the server and keep it running forever
    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # This will run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("WebSocket server stopped.")