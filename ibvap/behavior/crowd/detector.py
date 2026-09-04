"""
Crowd Gathering & Spatial Density Detection Subsystem.
Detects abnormal crowd gatherings, density surges, and localized congregation hotspots.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import uuid
import math

from ibvap.core.types import Track, AnalyticsEvent, EventType
from ibvap.core.config import IBVAPConfig, default_config


@dataclass
class CrowdCluster:
    """
    Represents a localized cluster of individuals.
    """
    cluster_id: str
    centroid: Tuple[int, int]
    count: int
    track_ids: List[int]
    bbox: Tuple[int, int, int, int]  # Enclosing bounding box (x1, y1, x2, y2)


class CrowdDetector:
    """
    Analyzes spatial proximity between human tracks to identify localized gatherings.
    """

    def __init__(
        self,
        config: Optional[IBVAPConfig] = None,
        crowd_threshold: int = 4,
        cluster_radius_px: float = 160.0,
    ):
        self.config = config or default_config
        self.crowd_threshold = crowd_threshold
        self.cluster_radius_px = cluster_radius_px
        self._last_alert_time: Dict[str, float] = {}  # camera_id -> timestamp

    def detect_clusters(self, person_tracks: List[Track]) -> List[CrowdCluster]:
        """
        Clusters person tracks based on spatial Euclidean proximity.
        """
        if len(person_tracks) < self.crowd_threshold:
            return []

        # Graph connected-components clustering
        n = len(person_tracks)
        adj = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                p1 = person_tracks[i].center
                p2 = person_tracks[j].center
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if dist <= self.cluster_radius_px:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        clusters: List[CrowdCluster] = []

        for i in range(n):
            if visited[i]:
                continue
            # BFS to find component
            queue = [i]
            visited[i] = True
            component = []
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            if len(component) >= self.crowd_threshold:
                tids = [person_tracks[idx].track_id for idx in component]
                xs = [person_tracks[idx].center[0] for idx in component]
                ys = [person_tracks[idx].center[1] for idx in component]
                centroid = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))

                # Calculate bounding box encompassing all cluster members
                x1s = [person_tracks[idx].bbox[0] for idx in component]
                y1s = [person_tracks[idx].bbox[1] for idx in component]
                x2s = [person_tracks[idx].bbox[2] for idx in component]
                y2s = [person_tracks[idx].bbox[3] for idx in component]
                enclosing_bbox = (min(x1s), min(y1s), max(x2s), max(y2s))

                cluster = CrowdCluster(
                    cluster_id=str(uuid.uuid4())[:8],
                    centroid=centroid,
                    count=len(component),
                    track_ids=tids,
                    bbox=enclosing_bbox,
                )
                clusters.append(cluster)

        return clusters

    def process(
        self,
        tracks: List[Track],
        camera_id: str = "camera-01",
        timestamp: Optional[float] = None,
    ) -> List[AnalyticsEvent]:
        """
        Evaluates active tracks for crowd gathering behavior.
        """
        now = timestamp if timestamp is not None else time.time()
        events: List[AnalyticsEvent] = []

        person_tracks = [t for t in tracks if t.class_name.lower() == "person"]
        clusters = self.detect_clusters(person_tracks)

        for cluster in clusters:
            last_alert = self._last_alert_time.get(camera_id, -1e9)
            if now - last_alert >= self.config.event_cooldown_seconds:
                event = AnalyticsEvent(
                    event_id=str(uuid.uuid4()),
                    camera_id=camera_id,
                    timestamp=now,
                    event_type=EventType.CROWD_GATHERING,
                    track_id=None,
                    identity_id=None,
                    confidence=min(1.0, 0.70 + 0.05 * cluster.count),
                    metadata={
                        "rule": "crowd_gathering",
                        "sub_type": "CROWD_GATHERING",
                        "cluster_id": cluster.cluster_id,
                        "headcount": cluster.count,
                        "threshold": self.crowd_threshold,
                        "centroid": cluster.centroid,
                        "bbox": cluster.bbox,
                        "involved_tracks": cluster.track_ids,
                    },
                )
                events.append(event)
                self._last_alert_time[camera_id] = now

        return events
