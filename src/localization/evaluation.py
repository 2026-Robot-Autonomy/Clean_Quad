"""Ground-truth evaluation, kept separate from runtime localization/fusion."""

import math

import numpy as np

from src.perception.detection import match_image


DISTANCE_BINS = ((0.0, 1.0, "<1m"), (1.0, 2.0, "1-2m"), (2.0, math.inf, ">=2m"))


def error_summary(rows):
    valid = [r for r in rows if r.get("error_cm") is not None]
    errors = [r["error_cm"] for r in valid]
    xy = [r["xy_error_cm"] for r in valid]
    return {
        "n": len(errors),
        "mean_error_cm": float(np.mean(errors)) if errors else None,
        "p95_error_cm": float(np.percentile(errors, 95)) if errors else None,
        "max_error_cm": float(np.max(errors)) if errors else None,
        "mean_xy_error_cm": float(np.mean(xy)) if xy else None,
    }


def _point_matches(tracks, truths, radius):
    """Maximum-cardinality class-aware matching inside an evaluation gate.

    Augmenting paths prevent a greedy match from counting an avoidable miss
    when two predictions can match one object and only one can match another.
    Candidate edges are searched from nearest to farthest.
    """

    candidates = {}
    for i, track in enumerate(tracks):
        edges = []
        for j, truth in enumerate(truths):
            if track.class_id == truth["class_id"]:
                distance = float(np.linalg.norm(track.position - truth["position"]))
                if distance <= radius:
                    edges.append((distance, j))
        candidates[i] = [j for _, j in sorted(edges)]
    owner = {}

    def assign(i, visited):
        for j in candidates[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in owner or assign(owner[j], visited):
                owner[j] = i
                return True
        return False

    for i in sorted(candidates, key=lambda i: (len(candidates[i]), tracks[i].track_id)):
        assign(i, set())
    return {i: j for j, i in owner.items()}, candidates


class LocalizationEvaluator:
    def __init__(self, objects, match_radius=0.25, target_error_cm=10.0):
        if not all(math.isfinite(v) and v > 0 for v in (match_radius, target_error_cm)):
            raise ValueError("evaluation distances must be positive and finite")
        self.objects = {o["name"]: o for o in objects}
        self.match_radius = match_radius
        self.target_error_cm = target_error_cm
        self.visible = set()
        self.rows = []
        self.detector_true_positives = 0
        self.localized_true_positives = 0

    def observe(self, frame, detections, truth_boxes, result):
        self.visible.update(b.name for b in truth_boxes)
        flags, matches = match_image(detections, [(b.class_id, b.box) for b in truth_boxes])
        self.detector_true_positives += sum(flags)
        for o in result.observations:
            index = matches[o.source_index]
            truth = truth_boxes[index] if index >= 0 else None
            delta = o.position - self.objects[truth.name]["position"] if truth else None
            self.localized_true_positives += truth is not None
            self.rows.append({
                "time": o.time, "detection_index": o.source_index,
                "track_id": result.assignments[o.source_index], "class_id": o.class_id,
                "score": o.score, "x": float(o.position[0]),
                "y": float(o.position[1]), "z": float(o.position[2]),
                "camera_distance": o.distance, "depth_pixels": o.depth_pixels,
                "depth_mad": o.depth_mad, "truth_name": truth.name if truth else "",
                "truth_distance": truth.distance if truth else None,
                "error_cm": float(np.linalg.norm(delta) * 100) if truth else None,
                "xy_error_cm": float(np.linalg.norm(delta[:2]) * 100) if truth else None,
            })

    def summarize(self, tracks):
        truths = [self.objects[name] for name in sorted(self.visible)]
        matches, candidates = _point_matches(tracks, truths, self.match_radius)
        map_rows = []
        for i, j in sorted(matches.items()):
            track, truth = tracks[i], truths[j]
            delta = track.position - truth["position"]
            map_rows.append({
                "track_id": track.track_id, "truth_name": truth["name"],
                "class_id": track.class_id,
                "error_cm": float(np.linalg.norm(delta) * 100),
                "xy_error_cm": float(np.linalg.norm(delta[:2]) * 100),
            })
        duplicates = sum(i not in matches and bool(candidates[i]) for i in range(len(tracks)))
        false_positives = sum(not candidates[i] for i in range(len(tracks)))
        found = {truths[j]["name"] for j in matches.values()}
        missed = sorted(self.visible - found)
        map_errors = error_summary(map_rows)
        passed = (
            bool(truths) and not missed and not duplicates and not false_positives
            and map_errors["max_error_cm"] <= self.target_error_cm
        )
        metrics = {
            "per_observation": error_summary(self.rows),
            "by_distance": {
                label: error_summary([
                    r for r in self.rows if r["truth_distance"] is not None
                    and lo <= r["truth_distance"] < hi
                ]) for lo, hi, label in DISTANCE_BINS
            },
            "detector_true_positives": int(self.detector_true_positives),
            "localized_true_positives": int(self.localized_true_positives),
            "localization_acceptance_rate": self.localized_true_positives / max(self.detector_true_positives, 1),
            "map": {
                **map_errors, "total_items": len(self.objects), "visible_items": len(truths),
                "not_visible_items": len(self.objects) - len(truths), "matched_items": len(matches),
                "confirmed_tracks": len(tracks), "duplicates": int(duplicates),
                "duplicate_rate": duplicates / max(len(tracks), 1),
                "false_positives": int(false_positives), "missed_items": len(missed),
                "missed_item_rate": len(missed) / max(len(truths), 1),
                "missed_names": missed, "visible_names": sorted(self.visible),
                "match_radius_m": self.match_radius, "target_error_cm": self.target_error_cm,
                "passed": bool(passed),
            },
        }
        return metrics, map_rows
