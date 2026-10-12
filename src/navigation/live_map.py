"""Explicit identity reconciliation from fresh online tracks to a prior map.

The new mission has a new run namespace. Stage 5 records are imported as prior
knowledge, with their source namespace recorded. Fresh tracker IDs are mapped
to mission-owned map IDs by class and position; numeric ID coincidence is never
used as association evidence.
"""

from copy import deepcopy
import math

import numpy as np

from src.mapping import SemanticMap


class LiveMapBridge:
    def __init__(self, prior, layout, *, run_id, association_radius=.20):
        if not math.isfinite(association_radius) or association_radius <= 0:
            raise ValueError("association radius must be positive and finite")
        if run_id == prior.run_id:
            raise ValueError("fresh tracker mission requires a new run namespace")
        self.map = SemanticMap.from_layout(layout, seed=prior.seed, run_id=run_id,
                                            resolution=prior.grid.resolution,
                                            provenance={"prior_map_run_id": prior.run_id,
                                                        "prior_map_provenance": deepcopy(prior.provenance),
                                                        "identity_policy": "explicit prior-map import, then online spatial/class reconciliation"})
        self.map.update_tracks(prior.source_records, seed=self.map.seed, run_id=self.map.run_id)
        for item in prior.items.values():
            if item["collected"]:
                self.map.mark_collected(item["track_id"])
        self.association_radius = association_radius
        self.aliases = {}
        self.updates = 0
        self.records_updated = 0
        self.new_items = 0
        self.outside_tracks = 0
        self.association_events = []

    def update(self, tracks):
        confirmed = [track for track in tracks if track.confirmed]
        records, used = [], set()
        aliases = dict(self.aliases)
        reserved = {aliases[track.track_id] for track in confirmed if track.track_id in aliases}
        events = []
        new_items = outside = 0
        for track in sorted(confirmed, key=lambda t: t.track_id):
            record = track.as_dict()
            try:
                self.map.grid.world_to_cell(*record["position"][:2])
            except IndexError:
                outside += 1
                continue
            online_id = record["track_id"]
            map_id = aliases.get(online_id)
            if map_id is None:
                candidates = []
                for item in self.map.items.values():
                    distance = float(np.linalg.norm(np.asarray(item["position"]) - record["position"]))
                    if item["track_id"] not in used | reserved and item["class_id"] == record["class_id"] and distance <= self.association_radius:
                        candidates.append((distance, item["track_id"]))
                if candidates:
                    map_id = min(candidates)[1]
                    association = "matched_prior_or_existing_map_record"
                else:
                    map_id = max([0] + list(self.map.items) + list(aliases.values())) + 1
                    new_items += 1
                    association = "new_online_map_record"
                aliases[online_id] = map_id
                events.append({"online_track_id": online_id, "map_track_id": map_id,
                               "association": association, "position": record["position"]})
            if map_id in used:
                raise ValueError("two online tracks cannot update the same map identity in one batch")
            used.add(map_id)
            record["track_id"] = map_id
            record["online_track_id"] = online_id
            record["online_run_id"] = self.map.run_id
            records.append(record)
        self.map.update_tracks(records, seed=self.map.seed, run_id=self.map.run_id)
        self.aliases = aliases
        self.association_events.extend(events)
        self.new_items += new_items
        self.outside_tracks += outside
        self.updates += 1
        self.records_updated += len(records)
        return len(records)

    def summary(self):
        return {"frame_batches": self.updates, "records_updated": self.records_updated,
                "online_identities": len(self.aliases), "new_items": self.new_items,
                "outside_map_track_samples": self.outside_tracks,
                "association_events": self.association_events, "run_id": self.map.run_id}
