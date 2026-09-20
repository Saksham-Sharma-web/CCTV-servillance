"""
IBVAP Unknown Persons & Re-ID Database Inspector.
Prints full details of all tracked unknown individuals, biometric embeddings,
camera transition paths, and sightings history.
"""

import os
import sys
import json
import sqlite3
import numpy as np

# Resolve database path relative to this script or current working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_PATHS = [
    os.path.join(SCRIPT_DIR, "data", "unknown_persons.db"),
    os.path.join(SCRIPT_DIR, "ibvap_rust", "data", "unknown_persons.db"),
    os.path.join(os.getcwd(), "data", "unknown_persons.db"),
    os.path.join(os.getcwd(), "..", "data", "unknown_persons.db"),
]

db_path = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        db_path = p
        break

if not db_path:
    db_path = os.path.join(SCRIPT_DIR, "data", "unknown_persons.db")

print(f"\n=======================================================")
print(f"📁 DATABASE: {os.path.abspath(db_path)}")
print(f"=======================================================\n")

if not os.path.exists(db_path):
    print("❌ Database file does not exist yet. Run the cameras to start tracking!")
    sys.exit(0)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# ── 1. Unknown Identities ──
cursor.execute("SELECT * FROM unknown_persons ORDER BY last_seen_timestamp DESC")
persons = cursor.fetchall()
print(f"👤 UNKNOWN IDENTITIES REGISTERED: {len(persons)}")
print("-" * 55)

for p in persons:
    uid = p["unknown_id"]
    first_cam = p["first_camera_id"]
    last_cam = p["last_camera_id"]
    cam_seq = json.loads(p["camera_sequence"]) if p["camera_sequence"] else []
    meta = json.loads(p["metadata"]) if p["metadata"] else {}
    
    # Biometric embeddings
    proto_raw = p["prototype_embedding"]
    proto_emb = np.frombuffer(proto_raw, dtype=np.float32) if proto_raw else np.array([])
    rep_raw = p["representative_embeddings"]
    rep_count = len(rep_raw) // (512 * 4) if rep_raw else 0
    
    print(f"• ID                  : {uid}")
    print(f"  First Seen Camera   : {first_cam}")
    print(f"  Last Seen Camera    : {last_cam}")
    print(f"  Camera Path         : {' ➔ '.join(cam_seq)}")
    print(f"  First Registered At : {p['created_at_iso']}")
    print(f"  Last Updated At     : {p['updated_at_iso']}")
    print(f"  Biometric Embedding : 512-D float32 vector (L2 norm: {np.linalg.norm(proto_emb):.4f})")
    print(f"  Embedding Sample    : {np.round(proto_emb[:6], 4).tolist()} ...")
    print(f"  Representative Views: {rep_count} additional angle(s)/embedding(s)")
    print(f"  Metadata            : {meta}")
    print("-" * 55)

# ── 2. Sightings Log ──
cursor.execute("SELECT * FROM unknown_person_sightings ORDER BY timestamp ASC")
sightings = cursor.fetchall()
print(f"\n👁️  SIGHTINGS LOG (CHRONOLOGICAL): {len(sightings)}")
print("-" * 55)

for s in sightings:
    bbox = json.loads(s["bbox"]) if s["bbox"] else []
    meta = json.loads(s["metadata"]) if s["metadata"] else {}
    print(f"• Sighting ID : {s['sighting_id']}")
    print(f"  Person ID   : {s['unknown_id']}")
    print(f"  Camera ID   : {s['camera_id']}")
    print(f"  Track ID    : {s['track_id']}")
    print(f"  Time (ISO)  : {s['timestamp_iso']}")
    print(f"  Similarity  : {s['similarity']:.4f}")
    print(f"  Face Quality: {s['face_quality']:.2f}")
    print(f"  Bounding Box: {bbox}")
    if meta:
        print(f"  Details     : {meta}")
    print("-" * 55)
