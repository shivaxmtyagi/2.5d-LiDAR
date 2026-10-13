"""
Temporal object tracking (Module 6) — nearest-neighbor centroid association.

This is the mechanism that actually decides static vs. dynamic: a single
frame's geometry can't tell you whether a box-shaped cluster is a parked car
or a moving one. Across frames, if a track's centroid keeps moving faster
than `dynamic_speed_threshold`, it gets marked dynamic; a consistently
still track is static; unmatched tracks are aged out after
`max_missed_frames` (this also drives the temporal map decay in
src/mapping/temporal.py, per spec section 22: stale moving objects must not
persist forever).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.objects.clustering import DetectedObject


@dataclass
class Track:
    track_id: int
    last_centroid: np.ndarray
    last_timestamp: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    missed_frames: int = 0
    hits: int = 1
    state: str = "static_obstacle"  # "static_obstacle" | "dynamic_object"


class ObjectTracker:
    def __init__(self, max_match_distance: float = 1.5, dynamic_speed_threshold: float = 0.3,
                 max_missed_frames: int = 5):
        self.max_match_distance = max_match_distance
        self.dynamic_speed_threshold = dynamic_speed_threshold
        self.max_missed_frames = max_missed_frames
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[DetectedObject], timestamp: float) -> List[DetectedObject]:
        """Associate detections with existing tracks (greedy nearest-neighbor
        in xy), update velocity/state, age out unmatched tracks, and return
        the same detections annotated with track_id / predicted_class /
        velocity."""
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = list(range(len(detections)))

        # Greedy nearest-neighbor matching, cheapest reasonable approach for
        # a small number of obstacles per frame (see README for a note on
        # upgrading to a Hungarian/Kalman tracker for denser scenes).
        pairs = []
        for tid in unmatched_tracks:
            track = self.tracks[tid]
            for di in unmatched_dets:
                dist = float(np.linalg.norm(detections[di].centroid[:2] - track.last_centroid[:2]))
                if dist <= self.max_match_distance:
                    pairs.append((dist, tid, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks, matched_dets = set(), set()
        for dist, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)
            track = self.tracks[tid]
            dt = max(timestamp - track.last_timestamp, 1e-6)
            displacement = detections[di].centroid[:2] - track.last_centroid[:2]
            velocity = displacement / dt
            speed = float(np.linalg.norm(velocity))

            track.velocity = velocity.astype(np.float32)
            track.last_centroid = detections[di].centroid
            track.last_timestamp = timestamp
            track.missed_frames = 0
            track.hits += 1
            if speed >= self.dynamic_speed_threshold:
                track.state = "dynamic_object"
            elif track.hits >= 3 and speed < self.dynamic_speed_threshold:
                track.state = "static_obstacle"

            detections[di].track_id = tid
            detections[di].predicted_class = track.state
            detections[di].velocity = track.velocity
            detections[di].confidence = min(0.5 + 0.1 * track.hits, 0.95)

        unmatched_tracks -= matched_tracks
        unmatched_dets = [d for d in unmatched_dets if d not in matched_dets]

        # New tracks for unmatched detections.
        for di in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(track_id=tid, last_centroid=detections[di].centroid, last_timestamp=timestamp)
            detections[di].track_id = tid
            detections[di].predicted_class = "static_obstacle"  # innocent until proven moving
            detections[di].confidence = 0.4

        # Age out tracks that went unmatched this frame.
        for tid in unmatched_tracks:
            self.tracks[tid].missed_frames += 1
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed_frames > self.max_missed_frames:
                del self.tracks[tid]

        return detections
