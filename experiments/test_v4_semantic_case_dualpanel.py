#!/usr/bin/env python
"""Lightweight checks for V4 semantic-case dual-panel visualization helpers."""

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from visualization.viz_v4_semantic_case_dualpanel import (  # noqa: E402
    aggregate_active,
    build_node_sample_metadata,
    case_direction_metrics,
    case_event_label,
    case_is_eligible,
    case_selection_priority,
    daytime_fraction,
    display_model_name,
    interval_anomaly_ratio,
    interval_sample_range,
    pick_representative_nodes,
    pool_node_embeddings_to_samples,
    select_case_points,
    select_representative_node_for_case,
    summarize_cluster_patterns,
    transition_interval_for_run,
)


class V4HelperTest(unittest.TestCase):
    def test_interval_sample_range_covers_step0_target_interval(self):
        i_lo, i_hi = interval_sample_range(
            r_start=120,
            r_end=130,
            te_s=100,
            te_e=180,
            horizon=12,
            pred_len=12,
        )
        self.assertEqual(i_lo, 108)
        self.assertEqual(i_hi, 118)

    def test_aggregate_active_reduces_last_dimension_only(self):
        arr = np.array([
            [[1.0, 10.0, 3.0], [5.0, 20.0, 7.0]],
            [[2.0, 30.0, 4.0], [6.0, 40.0, 8.0]],
        ])
        active = np.array([True, False, True])
        out = aggregate_active(arr, active)
        expected = np.array([[2.0, 6.0], [3.0, 7.0]])
        np.testing.assert_allclose(out, expected)

    def test_build_node_sample_metadata_matches_encode_row_order(self):
        meta = build_node_sample_metadata(
            sample_starts=np.array([100, 101]),
            node_count=3,
            horizon=12,
        )
        self.assertEqual(meta[0]["sample_start"], 100)
        self.assertEqual(meta[0]["node"], 0)
        self.assertEqual(meta[0]["target_start"], 112)
        self.assertEqual(meta[3]["sample_start"], 101)
        self.assertEqual(meta[3]["node"], 0)
        self.assertEqual(meta[5]["node"], 2)

    def test_pick_representative_nodes_prefers_active_nodes_with_best_gain(self):
        errors = {
            "STCross": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            "LSTM": np.array([[5.0, 4.0, 10.0], [5.0, 4.0, 10.0]]),
            "DAN": np.array([[4.0, 6.0, 8.0], [4.0, 6.0, 8.0]]),
        }
        active = np.array([True, False, True])
        selected = pick_representative_nodes(errors, active, max_nodes=1)
        self.assertEqual(selected.tolist(), [2])

    def test_select_case_points_finds_labelled_nodes_inside_runs(self):
        metadata = [
            {"sample_start": 10, "target_start": 22, "node": 0},
            {"sample_start": 11, "target_start": 23, "node": 1},
            {"sample_start": 40, "target_start": 52, "node": 0},
            {"sample_start": 41, "target_start": 53, "node": 1},
        ]
        selected = select_case_points(
            metadata=metadata,
            holiday_run=(22, 24),
            weather_run=(52, 54),
            holiday_nodes=np.array([1]),
            weather_nodes=np.array([0]),
        )
        self.assertEqual(selected["Holiday"], 1)
        self.assertEqual(selected["Weather"], 2)

    def test_pool_node_embeddings_to_samples_uses_active_nodes(self):
        z_node = np.array([
            [1.0, 1.0], [10.0, 10.0], [3.0, 5.0],
            [2.0, 4.0], [20.0, 20.0], [6.0, 8.0],
        ])
        pooled = pool_node_embeddings_to_samples(
            z_node=z_node,
            sample_count=2,
            node_count=3,
            active=np.array([True, False, True]),
        )
        expected = np.array([
            [2.0, 3.0, 1.0, 2.0],
            [4.0, 6.0, 2.0, 2.0],
        ])
        np.testing.assert_allclose(pooled, expected)

    def test_summarize_cluster_patterns_averages_active_node_deviation(self):
        delta = np.array([
            [[1.0, 10.0], [3.0, 30.0]],
            [[5.0, 50.0], [7.0, 70.0]],
            [[2.0, 20.0], [4.0, 40.0]],
        ])
        patterns = summarize_cluster_patterns(
            delta_windows=delta,
            cluster_labels=np.array([0, 1, 0]),
            active=np.array([True, False]),
            target_clusters=[0, 1],
        )
        np.testing.assert_allclose(patterns[0], np.array([1.5, 3.5]))
        np.testing.assert_allclose(patterns[1], np.array([5.0, 7.0]))

    def test_display_model_name_renames_our_model(self):
        self.assertEqual(display_model_name("STCross"), "ProtoDRIFT")
        self.assertEqual(display_model_name("ModernTCN"), "ModernTCN")

    def test_interval_anomaly_ratio_counts_half_transition(self):
        mask = np.array([False, False, False, True, True, True])
        self.assertEqual(interval_anomaly_ratio(mask, 0, 5), 0.5)

    def test_daytime_fraction_penalizes_midnight_cases(self):
        timestamps = np.array([
            np.datetime64("2017-01-01T00"),
            np.datetime64("2017-01-01T01"),
            np.datetime64("2017-01-01T12"),
            np.datetime64("2017-01-01T13"),
        ])
        self.assertEqual(daytime_fraction(timestamps, 0, 1), 0.0)
        self.assertEqual(daytime_fraction(timestamps, 2, 3), 1.0)

    def test_transition_interval_for_run_prefers_half_anomaly_daytime(self):
        mask = np.zeros(30, dtype=bool)
        mask[12:20] = True
        timestamps = np.array([
            np.datetime64("2017-01-01T00") + np.timedelta64(i, "h")
            for i in range(30)
        ])
        interval = transition_interval_for_run(
            run=(12, 19),
            mask=mask,
            timestamps=timestamps,
            lo=0,
            hi=30,
            window_len=12,
        )
        self.assertEqual(interval, (6, 17))
        self.assertAlmostEqual(interval_anomaly_ratio(mask, *interval), 0.5)

    def test_transition_interval_for_run_avoids_midnight_when_half_is_impossible(self):
        mask = np.zeros(30, dtype=bool)
        mask[0:24] = True
        timestamps = np.array([
            np.datetime64("2017-01-01T00") + np.timedelta64(i, "h")
            for i in range(30)
        ])
        interval = transition_interval_for_run(
            run=(0, 23),
            mask=mask,
            timestamps=timestamps,
            lo=0,
            hi=30,
            window_len=12,
        )
        self.assertEqual(interval, (6, 17))
        self.assertEqual(daytime_fraction(timestamps, *interval), 1.0)
    def test_case_direction_metrics_prefers_holiday_surge_underestimated_by_baselines(self):
        true_s0 = np.array([[10.0, 20.0], [12.0, 22.0]])
        nbl_s0 = np.array([[8.0, 18.0], [9.0, 19.0]])
        baseline_preds = {
            "ModernTCN": np.array([[7.0, 17.0], [8.0, 18.0]]),
            "DAN": np.array([[9.0, 19.0], [10.0, 20.0]]),
        }
        metrics = case_direction_metrics(
            true_s0=true_s0,
            nbl_s0=nbl_s0,
            baseline_preds=baseline_preds,
            active=np.array([True, True]),
            label="Holiday",
        )
        self.assertGreater(metrics["flow_delta"], 0.0)
        self.assertLess(metrics["baseline_bias"], 0.0)
        self.assertTrue(metrics["direction_ok"])

    def test_case_direction_metrics_prefers_weather_drop_overestimated_by_baselines(self):
        true_s0 = np.array([[5.0, 7.0], [4.0, 6.0]])
        nbl_s0 = np.array([[8.0, 10.0], [9.0, 11.0]])
        baseline_preds = {
            "ModernTCN": np.array([[9.0, 11.0], [10.0, 12.0]]),
            "DAN": np.array([[7.0, 9.0], [8.0, 10.0]]),
        }
        metrics = case_direction_metrics(
            true_s0=true_s0,
            nbl_s0=nbl_s0,
            baseline_preds=baseline_preds,
            active=np.array([True, True]),
            label="Weather",
        )
        self.assertLess(metrics["flow_delta"], 0.0)
        self.assertGreater(metrics["baseline_bias"], 0.0)
        self.assertTrue(metrics["direction_ok"])

    def test_case_event_label_names_known_test_holidays_and_rainstorm(self):
        self.assertEqual(case_event_label("Holiday", np.datetime64("2017-02-20T09")),
                         "Presidents' Day")
        self.assertEqual(case_event_label("Weather", np.datetime64("2017-06-01T09")),
                         "Rainstorm")
    def test_case_selection_priority_prefers_holiday_positive_all_steps_and_close_stcross(self):
        all_positive = dict(
            label="Holiday", direction_ok=True, transition_bonus=0.5,
            direction_score=1.0, baseline_direction_score=1.0,
            positive_step_fraction=1.0, min_flow_delta=0.2,
            true_mean=12.0, mae_st=0.3, margin=0.5)
        partly_negative = dict(
            label="Holiday", direction_ok=True, transition_bonus=0.5,
            direction_score=1.5, baseline_direction_score=1.5,
            positive_step_fraction=0.8, min_flow_delta=-0.1,
            true_mean=12.0, mae_st=0.1, margin=0.7)
        self.assertGreater(
            case_selection_priority(all_positive),
            case_selection_priority(partly_negative),
        )

    def test_case_selection_priority_prefers_weather_with_nontrivial_flow(self):
        nontrivial_drop = dict(
            label="Weather", direction_ok=True, transition_bonus=0.5,
            direction_score=0.8, baseline_direction_score=0.8,
            positive_step_fraction=0.0, min_flow_delta=-1.0,
            true_mean=3.0, mae_st=0.25, margin=0.4)
        too_low = dict(
            label="Weather", direction_ok=True, transition_bonus=0.5,
            direction_score=2.0, baseline_direction_score=2.0,
            positive_step_fraction=0.0, min_flow_delta=-3.0,
            true_mean=0.1, mae_st=0.05, margin=0.6)
        self.assertGreater(
            case_selection_priority(nontrivial_drop),
            case_selection_priority(too_low),
        )
    def test_select_representative_node_for_weather_avoids_trivial_low_flow_node(self):
        case = {
            "true": np.array([[0.0, 2.0], [0.0, 2.2], [0.0, 1.8]]),
            "nbl": np.array([[1.0, 3.0], [1.0, 3.2], [1.0, 2.8]]),
            "preds": {
                "STCross": np.array([[0.0, 2.1], [0.0, 2.0], [0.0, 1.7]]),
                "ModernTCN": np.array([[2.0, 4.0], [2.0, 4.2], [2.0, 3.8]]),
            },
        }
        node = select_representative_node_for_case(
            case=case,
            active=np.array([True, True]),
            label="Weather",
        )
        self.assertEqual(node, 1)
    def test_select_representative_node_for_weather_prefers_nonnegative_stcross_curve(self):
        case = {
            "true": np.array([[3.0, 2.0], [3.2, 2.2], [2.8, 1.8]]),
            "nbl": np.array([[4.0, 3.0], [4.2, 3.2], [3.8, 2.8]]),
            "preds": {
                "STCross": np.array([[3.1, 2.1], [-0.2, 2.0], [2.9, 1.7]]),
                "ModernTCN": np.array([[5.0, 4.0], [5.2, 4.2], [4.8, 3.8]]),
            },
        }
        node = select_representative_node_for_case(
            case=case,
            active=np.array([True, True]),
            label="Weather",
        )
        self.assertEqual(node, 1)
    def test_case_selection_priority_ranks_holiday_surge_magnitude_first(self):
        strong_surge = dict(
            label="Holiday", direction_ok=True, transition_bonus=0.5,
            direction_score=1.0, baseline_direction_score=1.0,
            flow_direction_score=2.0,
            positive_step_fraction=1.0, min_flow_delta=0.2,
            true_mean=12.0, mae_st=0.5, margin=0.1)
        weak_surge_close_fit = dict(
            label="Holiday", direction_ok=True, transition_bonus=0.5,
            direction_score=0.4, baseline_direction_score=1.2,
            flow_direction_score=0.5,
            positive_step_fraction=1.0, min_flow_delta=0.1,
            true_mean=12.0, mae_st=0.1, margin=0.6)
        self.assertGreater(
            case_selection_priority(strong_surge),
            case_selection_priority(weak_surge_close_fit),
        )

    def test_case_selection_priority_ranks_weather_drop_magnitude_first(self):
        deep_drop = dict(
            label="Weather", direction_ok=True, transition_bonus=0.5,
            direction_score=1.0, baseline_direction_score=1.0,
            flow_direction_score=1.5,
            positive_step_fraction=0.0, min_flow_delta=-2.0,
            true_mean=3.0, mae_st=0.4, margin=0.1)
        shallow_drop_close_fit = dict(
            label="Weather", direction_ok=True, transition_bonus=0.5,
            direction_score=0.3, baseline_direction_score=1.2,
            flow_direction_score=0.3,
            positive_step_fraction=0.0, min_flow_delta=-0.5,
            true_mean=3.0, mae_st=0.05, margin=0.5)
        self.assertGreater(
            case_selection_priority(deep_drop),
            case_selection_priority(shallow_drop_close_fit),
        )

    def test_case_is_eligible_gates_on_flow_direction_not_baseline_bias(self):
        good_holiday = dict(label="Holiday", direction_ok=True,
                            flow_direction_score=2.0,
                            positive_step_fraction=1.0, true_mean=5.0)
        mixed_holiday = dict(label="Holiday", direction_ok=True,
                             flow_direction_score=1.0,
                             positive_step_fraction=0.4, true_mean=5.0)
        weather_no_baseline_overshoot = dict(label="Weather", direction_ok=False,
                                             flow_direction_score=0.9,
                                             positive_step_fraction=0.0, true_mean=2.0)
        low_weather = dict(label="Weather", direction_ok=True,
                           flow_direction_score=1.8,
                           positive_step_fraction=0.0, true_mean=0.3)
        self.assertTrue(case_is_eligible(good_holiday))
        self.assertFalse(case_is_eligible(mixed_holiday))
        self.assertTrue(case_is_eligible(weather_no_baseline_overshoot))
        self.assertFalse(case_is_eligible(low_weather))


if __name__ == "__main__":
    unittest.main()
