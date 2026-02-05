Refactor Plan: Include Current Playing Track (SegmentOut) in GET /v1/rooms
Goal
Augment GET /v1/rooms so each returned room includes the current playing track segment (SegmentOut) based on server-calculated playhead:

playhead_ms = server_now_epoch_ms - room.play_started_at_epoch_ms
Look up the room’s mix_id → mix’s current_ready_revision_no → that revision’s segments
Return the segment where start_ms <= playhead_ms < end_ms as segmentout
If no segment matches (mix ended / playhead past end), return an explicit “no current track” signal.
Non-goals:

Do not change WS payload behavior beyond any schema-side optional fields.
Do not change mix rendering logic.
A) Schema changes (response contract)
In room.py, add a list-only room output model:
RoomListItemOut (room fields + playback fields)
Fields:
segmentout: SegmentOut | None (nullable; null means no current playing track)
is_playing: bool (explicit flag; True iff segmentout is not None)
Update RoomListOut.rooms to be List[RoomListItemOut] (keep RoomOut unchanged for create/rename/WS).
B) Data access helpers (avoid duplicating query logic)
In mix_repo.py, add targeted queries:
get_current_ready_revision(mix_id) -> MixRevision | None (join mixes → mix_revisions via current_ready_revision_no)
get_segment_at_playhead(mix_revision_id, playhead_ms) -> (segment fields + item metadata) | None
SQL filter: start_ms <= playhead_ms AND end_ms > playhead_ms
Join mix_items for song_name/artist_name (via metadata_json)
Keep these helpers deterministic (stable ordering; limit 1).
C) Service-layer assembly (single “compute segmentout” function)
In room_service.py, add:
get_room_segmentout(room, *, server_now_epoch_ms: int) -> tuple[SegmentOut | None, bool]
Logic:
playhead_ms = max(0, server_now_epoch_ms - room.play_started_at_epoch_ms)
Fetch current ready revision; if missing → return (None, False)
Fetch segment at playhead; if missing → return (None, False)
Build SegmentOut from segment row + metadata; return (segmentout, True)
Compute server_now_epoch_ms once per request (used for all rooms for consistency).
D) Endpoint wiring (GET /v1/rooms)
In rooms.py:
Update list_rooms to:
rooms = await svc.list_rooms()
now_ms = int(datetime.now(timezone.utc).timestamp() * 1000) (or reuse a shared _now_ms() helper)
For each room:
segmentout, is_playing = await svc.get_room_segmentout(room, server_now_epoch_ms=now_ms)
Build RoomListItemOut including segmentout + is_playing + existing room fields (and participant count)
Keep existing create_room, rename_room, delete_room behavior unchanged.
E) Docs + acceptance checks
Update swagger.yaml (or regenerate) so GET /v1/rooms shows the new per-room fields.
Acceptance:
While mix is mid-play, segmentout matches the segment containing computed playhead.
After mix ends, response includes segmentout: null and is_playing: false.
Deterministic: same DB state + same server_now_epoch_ms ⇒ same segmentout.
Quick manual validation:
Create a room, immediately call GET /v1/rooms and confirm is_playing=true with a non-null segmentout.
Temporarily simulate “mix ended” by setting play_started_at_epoch_ms far in the past (DB) and confirm segmentout=null.

