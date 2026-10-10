"""Spatial fusion and temporal class voting for stationary litter."""

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class LitterTrack:
    track_id: int
    class_id: int
    position: np.ndarray
    first_seen: float
    last_seen: float
    observations: int = 1
    score_sum: float = 0.0
    max_score: float = 0.0
    _weight: float = 0.0
    _m2: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)
    _class_weights: dict = field(default_factory=dict, repr=False)
    _class_counts: dict = field(default_factory=dict, repr=False)
    _last_position: np.ndarray = field(default=None, repr=False)
    confirmed: bool = False

    @classmethod
    def from_observation(cls, track_id, observation):
        return cls(
            track_id, observation.class_id, observation.position.copy(),
            observation.time, observation.time, score_sum=observation.score,
            max_score=observation.score, _weight=max(observation.score, 1e-6),
            _class_weights={observation.class_id: max(observation.score, 1e-6)},
            _class_counts={observation.class_id: 1},
            _last_position=observation.position.copy(),
        )

    def update(self, observation):
        weight = max(observation.score, 1e-6)
        total = self._weight + weight
        delta = observation.position - self.position
        self.position += delta * weight / total
        self._m2 += weight * delta * (observation.position - self.position)
        self._weight = total
        self.observations += 1
        self.last_seen = observation.time
        self._last_position = observation.position.copy()
        self.score_sum += observation.score
        self.max_score = max(self.max_score, observation.score)
        self._class_weights[observation.class_id] = self._class_weights.get(observation.class_id, 0.0) + weight
        self._class_counts[observation.class_id] = self._class_counts.get(observation.class_id, 0) + 1
        # A detector can briefly change its label for the same physical item.
        # Keep the strongest accumulated class; retain the current class on ties.
        self.class_id = max(self._class_weights, key=lambda c: (
            self._class_weights[c], c == self.class_id, -c,
        ))

    @property
    def mean_score(self):
        return self.score_sum / self.observations

    @property
    def class_observations(self):
        return self._class_counts.get(self.class_id, 0)

    @property
    def position_std(self):
        return np.sqrt(np.maximum(self._m2 / self._weight, 0.0))

    def as_dict(self):
        classes = ("bottle", "can", "cup", "wrapper")
        return {
            "track_id": self.track_id, "class_id": self.class_id,
            "class": classes[self.class_id], "position": self.position.tolist(),
            "position_std": self.position_std.tolist(),
            "observations": self.observations, "mean_score": self.mean_score,
            "class_observations": self.class_observations,
            "max_score": self.max_score, "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "class_confidence": self._class_weights[self.class_id] / self._weight,
            "class_evidence": {classes[c]: weight for c, weight in sorted(self._class_weights.items())},
        }


class LitterTracker:
    """A persistent map: confirmed stationary tracks do not expire.

    Each track can receive at most one observation in a frame. A new item is
    confirmed only after min_observations separate frames supporting its winning
    class. Recent surface positions aid association without changing the fused
    map estimate. Unconfirmed tracks expire after tentative_ttl seconds;
    reset() starts a new episode.
    """

    def __init__(self, association_radius=0.20, min_observations=3, tentative_ttl=3.0, class_switch_radius=0.08, association_window=1.0):
        if not math.isfinite(association_radius) or association_radius <= 0:
            raise ValueError("association_radius must be positive and finite")
        if not isinstance(min_observations, int) or min_observations < 1:
            raise ValueError("min_observations must be a positive integer")
        if not math.isfinite(tentative_ttl) or tentative_ttl <= 0:
            raise ValueError("tentative_ttl must be positive and finite")
        if not math.isfinite(class_switch_radius) or class_switch_radius <= 0:
            raise ValueError("class_switch_radius must be positive and finite")
        if not math.isfinite(association_window) or association_window <= 0:
            raise ValueError("association_window must be positive and finite")
        self.association_radius = association_radius
        self.min_observations = min_observations
        self.tentative_ttl = tentative_ttl
        self.class_switch_radius = min(class_switch_radius, association_radius)
        self.association_window = association_window
        self.reset()

    def reset(self):
        self.tracks = []
        self._next_id = 1
        self._last_time = -math.inf

    @property
    def confirmed_tracks(self):
        return [t for t in self.tracks if t.confirmed]

    def _confirm(self, track):
        # Three mixed labels must not turn one false detection followed by two
        # stolen observations into a permanent map entry. Confirmation persists.
        track.confirmed |= track.class_observations >= self.min_observations

    def update(self, observations, time):
        if not math.isfinite(time) or time <= self._last_time:
            raise ValueError("frame times must be finite and strictly increasing; reset for a new episode")
        observations = list(observations)
        for o in observations:
            if o.time != time or np.asarray(o.position).shape != (3,) or not np.all(np.isfinite(o.position)):
                raise ValueError("observations must have this frame's time and a finite 3-D position")
        self._last_time = time
        self.tracks = [
            t for t in self.tracks if t.confirmed
            or time - t.last_seen <= self.tentative_ttl
        ]
        pairs = []
        for i, observation in enumerate(observations):
            for j, track in enumerate(self.tracks):
                map_distance = float(np.linalg.norm(observation.position - track.position))
                # Changing viewpoints shift the measured surface relative to
                # the long-term mean. A fresh, consistent surface observation
                # helps preserve identity when a spurious nearby track appears.
                distance = map_distance
                if track._last_position is not None and time - track.last_seen <= self.association_window:
                    distance = min(distance, float(np.linalg.norm(observation.position - track._last_position)))
                gate = self.association_radius if observation.class_id == track.class_id else self.class_switch_radius
                if map_distance <= self.association_radius and distance <= gate:
                    pairs.append((distance, not track.confirmed, -observation.score, track.track_id, i, j))
        used_observations, used_tracks = set(), set()
        assignments = {}
        for _, _, _, _, i, j in sorted(pairs):
            if i in used_observations or j in used_tracks:
                continue
            self.tracks[j].update(observations[i])
            self._confirm(self.tracks[j])
            used_observations.add(i)
            used_tracks.add(j)
            assignments[observations[i].source_index] = self.tracks[j].track_id
        for i, observation in enumerate(observations):
            if i not in used_observations:
                track = LitterTrack.from_observation(self._next_id, observation)
                self._confirm(track)
                self._next_id += 1
                self.tracks.append(track)
                assignments[observation.source_index] = track.track_id
        return assignments
