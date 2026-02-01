import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from atomix.runtime.presence import presence

router = APIRouter()

@router.websocket("/rooms/{room_id}/ws")
async def ws_room(websocket: WebSocket, room_id: str):
    await websocket.accept()
    conn_id = str(uuid.uuid4())
    presence[room_id].add(conn_id)

    # Assign a server-side nickname for this connection
    your_name = f"user-{conn_id[:4]}"

    try:
        # TODO: load room snapshot from DB and send
        await websocket.send_json({"type": "room_snapshot", "your_name": your_name})
        while True:
            msg = await websocket.receive_json()
            # TODO: handle chat_send, etc.
    except WebSocketDisconnect:
        pass
    finally:
        presence[room_id].discard(conn_id)
        if not presence[room_id]:
            presence.pop(room_id, None)
