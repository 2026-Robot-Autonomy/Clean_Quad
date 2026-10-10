"""Analytic geometry, association, and evaluation regressions for Stage 4."""

from types import SimpleNamespace
import unittest

import numpy as np

from src.localization.evaluation import LocalizationEvaluator, _point_matches
from src.localization.geometry import LocalizationConfig, LocalizedDetection, localize, pixel_to_world
from src.localization.pipeline import LocalizationPipeline
from src.localization.tracking import LitterTrack, LitterTracker


def frame(time=0.0, depth=2.0, rotation=None):
    return SimpleNamespace(
        time=time, intrinsics=SimpleNamespace(width=100, height=80, fx=100., fy=100., cx=50., cy=40.),
        cam_pos=np.array([1., 2., 0.3]), cam_mat=np.eye(3) if rotation is None else rotation,
        depth=np.full((80, 100), depth), geom_ids=np.zeros((80, 100)),
    )


def detection(box=(45., 35., 55., 45.), class_id=0, score=0.9):
    return SimpleNamespace(box=box, class_id=class_id, score=score)


def observation(position=(0., 0., 0.), time=0., class_id=0, index=0, score=0.9):
    return LocalizedDetection(index, class_id, score, np.array(position, dtype=float), time, 1., "depth")


class GeometryTests(unittest.TestCase):
    def test_camera_axes_and_axial_depth(self):
        f = frame()
        np.testing.assert_allclose(pixel_to_world(f, 60., 50., 2.), [1.2, 1.8, -1.7])
        self.assertGreater(np.linalg.norm(pixel_to_world(f, 60., 50., 2.) - f.cam_pos), 2.)

    def test_projection_round_trip_with_rotated_camera_and_arrays(self):
        theta = 0.7
        rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        f = frame(rotation=rotation)
        u, v, depth = np.array([10., 55., 75.]), np.array([20., 40., 60.]), np.array([1., 2., 3.])
        points = pixel_to_world(f, u, v, depth)
        p = (points - f.cam_pos) @ f.cam_mat
        np.testing.assert_allclose(50 + 100 * p[:, 0] / -p[:, 2], u)
        np.testing.assert_allclose(40 - 100 * p[:, 1] / -p[:, 2], v)
        np.testing.assert_allclose(-p[:, 2], depth)

    def test_central_median_rejects_background_and_invalid_samples(self):
        f = frame(depth=6.)
        f.depth[37:43, 47:53] = 2.
        f.depth[38, 48] = np.nan
        f.depth[39, 49] = np.inf
        f.depth[40, 50] = 0.
        o, reason = localize(f, detection())
        self.assertIsNone(reason)
        np.testing.assert_allclose(o.position, [1., 2., -1.7])
        self.assertEqual(o.depth_pixels, 33)

    def test_invalid_depth_missing_depth_and_bad_boxes(self):
        f = frame(depth=np.nan)
        self.assertEqual(localize(f, detection())[1], "depth_pixels")
        f.depth = None
        self.assertEqual(localize(f, detection())[1], "missing_depth")
        self.assertEqual(localize(frame(), detection(box=(20, 20, 10, 10)))[1], "box")
        self.assertEqual(localize(frame(), detection(box=(np.nan, 0, 1, 2)))[1], "box")
        self.assertEqual(localize(frame(), detection(box=(-10, -10, -1, -1)))[1], "box")

    def test_score_range_and_bin_size_rejection(self):
        self.assertEqual(localize(frame(), detection(score=.1))[1], "score")
        self.assertEqual(localize(frame(depth=20.), detection())[1], "depth_pixels")
        self.assertEqual(localize(frame(), detection(box=(20, 10, 80, 70)))[1], "extent")

    def test_flat_ground_uses_ray_without_depth(self):
        f = frame()
        f.depth = None
        o, reason = localize(f, detection(), method="ground_plane")
        self.assertIsNone(reason)
        np.testing.assert_allclose(o.position, [1., 1.9865, 0.])

    def test_ground_ray_above_horizon_is_rejected(self):
        f = frame(rotation=np.diag([1., 1., -1.]))
        self.assertEqual(localize(f, detection(), method="ground_plane")[1], "ground_ray")

    def test_geometry_is_independent_of_segmentation(self):
        f = frame()
        a, _ = localize(f, detection())
        f.geom_ids[:] = 999
        b, _ = localize(f, detection())
        np.testing.assert_array_equal(a.position, b.position)

    def test_nonfinite_configuration_cannot_disable_filters(self):
        for field in ("max_extent", "max_range", "ground_z", "depth_crop"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                LocalizationConfig(**{field: np.nan})


class TrackingTests(unittest.TestCase):
    def test_repeated_observations_confirm_one_weighted_track(self):
        tracker = LitterTracker(min_observations=3)
        for t, x in enumerate([0., .04, .02]):
            tracker.update([observation((x, 0, 0), float(t))], float(t))
        self.assertEqual(len(tracker.confirmed_tracks), 1)
        np.testing.assert_allclose(tracker.confirmed_tracks[0].position, [.02, 0, 0])
        self.assertGreater(tracker.confirmed_tracks[0].position_std[0], 0)

    def test_confidence_changes_fused_position(self):
        tracker = LitterTracker(min_observations=2)
        tracker.update([observation(score=.6)], 0.)
        tracker.update([observation((.1, 0, 0), 1., score=.9)], 1.)
        np.testing.assert_allclose(tracker.confirmed_tracks[0].position, [.06, 0, 0])

    def test_same_frame_cannot_confirm_or_collapse_two_items(self):
        tracker = LitterTracker()
        tracker.update([observation(index=0), observation((.05, 0, 0), index=1)], 0.)
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(len(tracker.confirmed_tracks), 0)
        tracker.update([observation((.01, 0, 0), 1., index=0), observation((.06, 0, 0), 1., index=1)], 1.)
        self.assertEqual([t.observations for t in tracker.tracks], [2, 2])

    def test_different_classes_outside_switch_gate_remain_separate(self):
        tracker = LitterTracker()
        tracker.update([observation()], 0.)
        tracker.update([observation((.12, 0, 0), time=1., class_id=1)], 1.)
        self.assertEqual(len(tracker.tracks), 2)

    def test_class_flicker_votes_into_one_physical_track(self):
        tracker = LitterTracker()
        labels = [1, 1, 2, 1, 2, 1]
        for t, label in enumerate(labels):
            tracker.update([observation((.01 * (t % 2), 0, 0), float(t), class_id=label)], float(t))
        self.assertEqual(len(tracker.confirmed_tracks), 1)
        track = tracker.confirmed_tracks[0]
        self.assertEqual(track.class_id, 1)
        self.assertEqual(track.observations, 6)
        self.assertAlmostEqual(track.as_dict()["class_confidence"], 4 / 6)

    def test_nearby_simultaneous_items_keep_separate_tracks_across_classes(self):
        tracker = LitterTracker()
        for t in range(3):
            tracker.update([
                observation(time=float(t), class_id=0, index=0),
                observation((.05, 0, 0), time=float(t), class_id=1, index=1),
            ], float(t))
        self.assertEqual(len(tracker.confirmed_tracks), 2)
        self.assertEqual([t.class_id for t in tracker.confirmed_tracks], [0, 1])

    def test_extra_wrapper_cannot_steal_established_bottle_identity(self):
        # Reproduce the failed CNN route: the long-term surface mean is east
        # of the current view, and a false wrapper appears beside the bottle.
        tracker = LitterTracker()
        for t in (0., .2, .4):
            tracker.update([observation((5.46, 3.88, .101), t)], t)
        tracker.update([observation((5.365684, 3.910749, .105143), 17.8)], 17.8)
        tracker.update([
            observation((5.368754, 3.929679, .100193), 18., score=.794184),
            observation((5.379165, 3.940365, .124434), 18., class_id=3,
                        index=1, score=.384542),
        ], 18.)
        for t, position, score in (
            (18.2, (5.369623, 3.935602, .095783), .641508),
            (18.4, (5.372788, 3.945503, .093529), .529096),
        ):
            self.assertEqual(tracker.update([observation(position, t, score=score)], t), {0: 1})
        tracker.update([], 22.)
        self.assertEqual([t.track_id for t in tracker.confirmed_tracks], [1])
        self.assertEqual(len(tracker.tracks), 1)

    def test_nearby_real_bottle_can_receive_observations_without_old_bottle(self):
        tracker = LitterTracker()
        for t in (0., .2, .4):
            tracker.update([observation(time=t)], t)
        tracker.update([
            observation(time=.6), observation((.05, 0, 0), .6, index=1),
        ], .6)
        for t in (.8, 1.):
            assignments = tracker.update([observation((.05, 0, 0), t)], t)
            self.assertEqual(assignments, {0: 2})
        self.assertEqual([t.track_id for t in tracker.confirmed_tracks], [1, 2])

    def test_confirmation_requires_repeat_support_for_winning_class(self):
        tracker = LitterTracker()
        for t, label, score in ((0., 3, .4), (.2, 0, .7), (.4, 0, .7)):
            tracker.update([observation(time=t, class_id=label, score=score)], t)
        self.assertEqual(tracker.confirmed_tracks, [])
        tracker.update([observation(time=.6, score=.7)], .6)
        self.assertEqual(len(tracker.confirmed_tracks), 1)
        self.assertEqual(tracker.confirmed_tracks[0].class_id, 0)

    def test_unconfirmed_mixed_labels_expire(self):
        tracker = LitterTracker()
        for t, label in ((0., 3), (.2, 0), (.4, 0)):
            tracker.update([observation(time=t, class_id=label)], t)
        tracker.update([], 4.)
        self.assertEqual(tracker.tracks, [])

    def test_old_surface_observation_does_not_override_map_after_long_gap(self):
        tracker = LitterTracker()
        for t in (0., .2, .4):
            tracker.update([observation(time=t)], t)
        tracker.update([observation((.18, 0, 0), .6)], .6)
        tracker.update([
            observation((.18, 0, 0), 10., index=0),
            observation((.045, 0, 0), 10., index=1),
        ], 10.)
        self.assertEqual(tracker.tracks[0].last_seen, 10.)
        self.assertLess(tracker.tracks[0].position[0], .06)

    def test_tentatives_expire_confirmed_tracks_persist_and_reset(self):
        tracker = LitterTracker(min_observations=2)
        tracker.update([observation(index=0), observation((1, 0, 0), index=1)], 0.)
        tracker.update([observation(time=1.)], 1.)
        tracker.update([], 10.)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(len(tracker.confirmed_tracks), 1)
        tracker.reset()
        self.assertEqual(tracker.tracks, [])
        tracker.update([observation()], 0.)
        self.assertEqual(tracker.tracks[0].track_id, 1)

    def test_time_replay_and_bad_positions_are_rejected(self):
        tracker = LitterTracker()
        tracker.update([], 0.)
        with self.assertRaises(ValueError):
            tracker.update([], 0.)
        with self.assertRaises(ValueError):
            tracker.update([observation((np.nan, 0, 0), 1.)], 1.)


class EvaluationTests(unittest.TestCase):
    def test_map_counts_duplicates_misses_false_positives_and_error(self):
        objects = [{"name": "a", "class_id": 0, "position": np.zeros(3)}, {"name": "b", "class_id": 1, "position": np.ones(3)}]
        evaluator = LocalizationEvaluator(objects)
        evaluator.visible.update(["a", "b"])
        tracks = [LitterTrack.from_observation(i + 1, observation(p)) for i, p in enumerate([(0.03, 0, 0), (.1, 0, 0), (2, 0, 0)])]
        metrics, rows = evaluator.summarize(tracks)
        self.assertEqual(metrics["map"]["matched_items"], 1)
        self.assertEqual(metrics["map"]["duplicates"], 1)
        self.assertEqual(metrics["map"]["false_positives"], 1)
        self.assertEqual(metrics["map"]["missed_names"], ["b"])
        self.assertAlmostEqual(rows[0]["error_cm"], 3.)
        self.assertFalse(metrics["map"]["passed"])

    def test_assignment_finds_maximum_cardinality(self):
        truths = [{"class_id": 0, "position": np.array(p)} for p in [(0., 0, 0), (.3, 0, 0)]]
        tracks = [LitterTrack.from_observation(1, observation((.1, 0, 0))), LitterTrack.from_observation(2, observation((-.1, 0, 0)))]
        matches, _ = _point_matches(tracks, truths, .25)
        self.assertEqual(len(matches), 2)

    def test_empty_map_does_not_certify_and_statistics_are_json_safe(self):
        metrics, rows = LocalizationEvaluator([]).summarize([])
        self.assertFalse(metrics["map"]["passed"])
        self.assertIsNone(metrics["map"]["mean_error_cm"])
        self.assertEqual(rows, [])

    def test_large_error_fails_even_when_all_items_are_matched(self):
        evaluator = LocalizationEvaluator([{"name": "a", "class_id": 0, "position": np.zeros(3)}])
        evaluator.visible.add("a")
        track = LitterTrack.from_observation(1, observation((.11, 0, 0)))
        metrics, _ = evaluator.summarize([track])
        self.assertEqual(metrics["map"]["missed_items"], 0)
        self.assertEqual(metrics["map"]["false_positives"], 0)
        self.assertFalse(metrics["map"]["passed"])

    def test_pipeline_preserves_detection_indices_after_rejection(self):
        pipeline = LocalizationPipeline()
        result = pipeline.update(frame(), [detection(score=.1), detection()])
        self.assertEqual(result.rejected, {0: "score"})
        self.assertEqual(result.observations[0].source_index, 1)
        self.assertEqual(set(result.assignments), {1})

    def test_competing_labels_on_overlapping_boxes_do_not_split_tracks(self):
        pipeline = LocalizationPipeline()
        result = pipeline.update(frame(), [detection(score=.9), detection(box=(47, 35, 57, 45), class_id=1, score=.6)])
        self.assertEqual(result.rejected, {1: "duplicate_box"})
        self.assertEqual(len(result.observations), 1)
        self.assertEqual(len(pipeline.tracker.tracks), 1)

    def test_separate_image_boxes_survive_duplicate_filter(self):
        pipeline = LocalizationPipeline()
        result = pipeline.update(frame(), [detection(), detection(box=(49, 35, 59, 45), class_id=1)])
        self.assertEqual(result.rejected, {})
        self.assertEqual(len(result.observations), 2)

    def test_nested_fragment_box_is_suppressed_despite_low_iou(self):
        pipeline = LocalizationPipeline()
        result = pipeline.update(frame(), [
            detection(), detection(box=(48, 38, 52, 42), class_id=3, score=.4),
        ])
        self.assertEqual(result.rejected, {1: "duplicate_box"})
        self.assertEqual(len(pipeline.tracker.tracks), 1)

    def test_nested_boxes_at_different_depths_remain_separate(self):
        f = frame()
        f.depth[38:42, 48:52] = 3.
        pipeline = LocalizationPipeline()
        result = pipeline.update(f, [
            detection(), detection(box=(48, 38, 52, 42), class_id=3, score=.4),
        ])
        self.assertEqual(result.rejected, {})
        self.assertEqual(len(pipeline.tracker.tracks), 2)


if __name__ == "__main__":
    unittest.main()
