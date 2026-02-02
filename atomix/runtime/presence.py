from collections import defaultdict
import uuid


# room_id -> set(connection_ids)
room_presence: defaultdict[uuid.UUID, set[str]] = defaultdict(set)