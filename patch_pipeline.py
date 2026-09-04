with open("ibvap/core/pipeline.py", "r") as f:
    code = f.read()

target = """                                    else:
                                        track.identity_id = None
                                        track.identity_name = "UNKNOWN PERSON"
"""

replacement = """                                    else:
                                        if track.identity_name != "UNKNOWN PERSON":
                                            candidate_events.append(
                                                AnalyticsEvent(
                                                    camera_id=camera_id,
                                                    timestamp=now,
                                                    event_type=EventType.UNKNOWN_PERSON,
                                                    track_id=track.track_id,
                                                    confidence=sim,
                                                    metadata={
                                                        "reason": "Face detected but no match in registry",
                                                        "track_id": track.track_id,
                                                        "similarity": round(sim, 4),
                                                    }
                                                )
                                            )
                                        track.identity_id = None
                                        track.identity_name = "UNKNOWN PERSON"
"""

if target in code:
    code = code.replace(target, replacement)
    with open("ibvap/core/pipeline.py", "w") as f:
        f.write(code)
    print("Patched successfully")
else:
    print("Target not found")
