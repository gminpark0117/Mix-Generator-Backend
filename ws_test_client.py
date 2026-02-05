#!/usr/bin/env python3
"""
WebSocket Test Client for Atomix API

Usage:
    python ws_test_client.py <server_url> <room_id>

Examples:
    python ws_test_client.py ws://localhost:8000 550e8400-e29b-41d4-a716-446655440000
    python ws_test_client.py ws://your-server.com 550e8400-e29b-41d4-a716-446655440000
    python ws_test_client.py wss://your-server.com 550e8400-e29b-41d4-a716-446655440000  # For HTTPS

Commands (while connected):
    - Type any message and press Enter to send a chat message
    - Type 'quit' or 'exit' to disconnect
    - Press Ctrl+C to force disconnect
"""

import asyncio
import json
import sys
from datetime import datetime

try:
    import websockets
except ImportError:
    print("Error: 'websockets' package not installed.")
    print("Install it with: pip install websockets")
    sys.exit(1)


def format_timestamp(ts: str | datetime) -> str:
    """Format a timestamp for display."""
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.strftime("%H:%M:%S")
        except:
            return ts
    return ts.strftime("%H:%M:%S")


def print_message(msg: dict) -> None:
    """Pretty print a received WebSocket message."""
    msg_type = msg.get("type", "unknown")
    
    print()
    print("=" * 60)
    
    if msg_type == "room_snapshot":
        print(f"📷 ROOM SNAPSHOT")
        print(f"   Your name: {msg.get('your_name', 'N/A')}")
        room = msg.get("room", {})
        print(f"   Room ID: {room.get('id', 'N/A')}")
        print(f"   Room Name: {room.get('name', 'N/A')}")
        print(f"   Participant Count: {room.get('participant_count', 'N/A')}")
        
        revision = msg.get("current_revision", {})
        print(f"   Current Revision: {revision.get('revision_num', 'N/A')}")
        
        tracklist = msg.get("tracklist", [])
        print(f"   Tracks in mix: {len(tracklist)}")
        
        chat_recent = msg.get("chat_recent", [])
        if chat_recent:
            print(f"   Recent chat messages ({len(chat_recent)}):")
            for chat in chat_recent[-5:]:  # Show last 5
                ts = format_timestamp(chat.get("created_at", ""))
                sender = chat.get("sender_name", "Unknown")
                message = chat.get("message", "")
                print(f"      [{ts}] {sender}: {message}")
    
    elif msg_type == "chat_message":
        ts = format_timestamp(msg.get("created_at", ""))
        sender = msg.get("sender_name", "Unknown")
        message = msg.get("message", "")
        seq = msg.get("seq", "?")
        print(f"💬 CHAT MESSAGE (#{seq})")
        print(f"   [{ts}] {sender}: {message}")
    
    elif msg_type == "participant_count_update":
        count = msg.get("participant_count", "?")
        print(f"👥 PARTICIPANT COUNT UPDATE: {count} participant(s)")
    
    elif msg_type == "revision_ready":
        revision = msg.get("revision", {})
        rev_num = revision.get("revision_num", "?")
        tracklist = msg.get("tracklist", [])
        print(f"🎵 REVISION READY: Revision #{rev_num}")
        print(f"   Tracks: {len(tracklist)}")
    
    else:
        print(f"📨 UNKNOWN MESSAGE TYPE: {msg_type}")
        print(f"   {json.dumps(msg, indent=2, default=str)}")
    
    print("=" * 60)


async def receive_messages(websocket) -> None:
    """Task to continuously receive and print messages."""
    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
                print_message(msg)
                print("\n[You] > ", end="", flush=True)
            except json.JSONDecodeError:
                print(f"\n⚠️  Received non-JSON message: {message}")
                print("\n[You] > ", end="", flush=True)
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n❌ Connection closed: {e.code} - {e.reason}")
    except Exception as e:
        print(f"\n❌ Error receiving message: {e}")


async def send_messages(websocket) -> None:
    """Task to read user input and send chat messages."""
    loop = asyncio.get_event_loop()
    
    print("\n✅ Connected! Type a message and press Enter to send.")
    print("   Type 'quit' or 'exit' to disconnect.\n")
    
    while True:
        try:
            # Read input in a non-blocking way
            print("[You] > ", end="", flush=True)
            user_input = await loop.run_in_executor(None, sys.stdin.readline)
            user_input = user_input.strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit"):
                print("👋 Disconnecting...")
                await websocket.close()
                break
            
            # Send chat message
            chat_msg = {
                "type": "chat_send",
                "message": user_input
            }
            await websocket.send(json.dumps(chat_msg))
            print(f"   ✓ Sent: {user_input}")
            
        except websockets.exceptions.ConnectionClosed:
            print("\n❌ Connection was closed")
            break
        except Exception as e:
            print(f"\n❌ Error sending message: {e}")
            break


async def main(server_url: str, room_id: str) -> None:
    """Main function to connect and handle WebSocket communication."""
    
    # Build WebSocket URL
    ws_url = f"{server_url}/v1/rooms/{room_id}/ws"
    
    print(f"🔌 Connecting to: {ws_url}")
    print("-" * 60)
    
    try:
        async with websockets.connect(ws_url) as websocket:
            # Create tasks for receiving and sending
            receive_task = asyncio.create_task(receive_messages(websocket))
            send_task = asyncio.create_task(send_messages(websocket))
            
            # Wait for either task to complete (usually send_task when user quits)
            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ Connection failed with status code: {e.status_code}")
        if e.status_code == 404:
            print("   Room not found. Make sure the room_id exists.")
        elif e.status_code == 403:
            print("   Access forbidden.")
        else:
            print(f"   Response: {e}")
    except websockets.exceptions.InvalidURI as e:
        print(f"❌ Invalid URI: {e}")
        print("   Make sure the server URL starts with ws:// or wss://")
    except ConnectionRefusedError:
        print("❌ Connection refused. Is the server running?")
    except Exception as e:
        print(f"❌ Connection error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        print("\nError: Missing arguments!")
        print(f"Usage: python {sys.argv[0]} <server_url> <room_id>")
        print(f"Example: python {sys.argv[0]} ws://localhost:8000 your-room-uuid-here")
        sys.exit(1)
    
    server_url = sys.argv[1].rstrip("/")
    room_id = sys.argv[2]
    
    # Validate URL scheme
    if not server_url.startswith(("ws://", "wss://")):
        print("⚠️  Warning: URL should start with ws:// or wss://")
        print(f"   Assuming ws:// prefix...")
        server_url = f"ws://{server_url}"
    
    try:
        asyncio.run(main(server_url, room_id))
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
