"""Fresh camera identities must never silently overwrite prior map IDs."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from src.environment.layout import default_layout
from src.mapping import SemanticMap
from src.navigation.live_map import LiveMapBridge


def record(identity, position=(1., 1., .05), class_id=0):
    name = ("bottle", "can", "cup", "wrapper")[class_id]
    return {"track_id": identity, "class_id": class_id, "class": name,
            "position": list(position), "position_std": [0., 0., 0.],
            "observations": 3, "class_observations": 3, "mean_score": .8,
            "max_score": .9, "first_seen": 0., "last_seen": 1.,
            "class_confidence": 1., "class_evidence": {name: 2.4}}


def track(value, confirmed=True):
    return SimpleNamespace(track_id=value["track_id"], confirmed=confirmed,
                           as_dict=lambda: deepcopy(value))


class LiveMapTests(unittest.TestCase):
    def setUp(self):
        self.layout = default_layout()
        self.prior = SemanticMap.from_layout(self.layout, seed=0, run_id="prior")
        self.prior.update_tracks([record(1), record(9, (2., 1., .05), 1)],
                                 seed=0, run_id="prior")
        self.bridge = LiveMapBridge(self.prior, self.layout, run_id="fresh")

    def test_numeric_id_coincidence_does_not_associate(self):
        self.bridge.update([track(record(1, (2.01, 1., .05), 1))])
        self.assertEqual(self.bridge.aliases[1], 9)
        self.assertEqual(self.bridge.map.item(1)["class"], "bottle")
        self.assertEqual(self.prior.item(9)["position"][0], 2.)

    def test_repeated_updates_preserve_collection_state_and_identity(self):
        self.prior.mark_collected(1)
        bridge = LiveMapBridge(self.prior, self.layout, run_id="fresh-collected")
        value = track(record(7, (1.02, 1., .05)))
        bridge.update([value])
        bridge.update([value])
        self.assertEqual(bridge.aliases, {7: 1})
        self.assertTrue(bridge.map.item(1)["collected"])
        self.assertEqual(len(bridge.map.items), 2)

    def test_unmatched_track_gets_new_mission_id(self):
        self.bridge.update([track(record(1, (3., 1., .05), 2))])
        self.assertEqual(self.bridge.aliases[1], 10)
        self.assertEqual(self.bridge.new_items, 1)
        self.assertEqual(self.bridge.map.item(10)["online_track_id"], 1)

    def test_existing_alias_is_reserved_before_new_association(self):
        self.bridge.update([track(record(8))])
        self.bridge.update([track(record(2)), track(record(8))])
        self.assertEqual(self.bridge.aliases[8], 1)
        self.assertNotEqual(self.bridge.aliases[2], 1)

    def test_invalid_batch_does_not_change_map_or_bridge(self):
        before = self.bridge.map.items
        invalid = record(2, (3., 1., .05))
        invalid["mean_score"] = 2.
        with self.assertRaises(ValueError):
            self.bridge.update([track(record(1)), track(invalid)])
        self.assertEqual(self.bridge.map.items, before)
        self.assertEqual(self.bridge.aliases, {})
        self.assertEqual(self.bridge.updates, 0)
        self.assertEqual(self.bridge.association_events, [])

    def test_outside_and_tentative_tracks_do_not_enter_map(self):
        count = self.bridge.update([track(record(2, (-1., 1., .05))),
                                    track(record(3), confirmed=False)])
        self.assertEqual(count, 0)
        self.assertEqual(self.bridge.outside_tracks, 1)
        self.assertEqual(self.bridge.aliases, {})

    def test_nonfinite_position_is_rejected_atomically(self):
        with self.assertRaises(ValueError):
            self.bridge.update([track(record(1, (float("nan"), 1., .05)))])
        self.assertEqual(self.bridge.updates, 0)

    def test_fresh_tracker_requires_fresh_namespace(self):
        with self.assertRaises(ValueError):
            LiveMapBridge(self.prior, self.layout, run_id="prior")


if __name__ == "__main__":
    unittest.main()
