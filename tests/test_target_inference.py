import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import opencommand
import target_inference


def sample_frame():
    rng = np.random.default_rng(7)
    rows = []
    for pitcher_id in (1, 2, 3):
        for _ in range(16):
            glove_x, glove_z = rng.normal(0, 2, 2)
            rows.append({
                "pitcher_id": pitcher_id,
                "pitch_type": "FF",
                "hand": "R",
                "naive_x_in": glove_x,
                "naive_z_in": 30 + glove_z,
                "plate_x_in": 1 + 0.5 * glove_x + rng.normal(),
                "plate_z_in": 34 + 0.5 * glove_z + rng.normal(),
            })
    return pd.DataFrame(rows)


class TargetInferenceTests(unittest.TestCase):
    def test_zero_precision_rows_do_not_change_random_effect_variance(self):
        estimate = pd.Series([0.0, 10.0])
        variance = pd.Series([1.0, 1.0])
        parent = pd.Series([5.0, 5.0])
        _, expected = target_inference.predict_random_effect(estimate, variance, parent)

        padding = 60
        padded_estimate = pd.concat([estimate, pd.Series(np.zeros(padding))], ignore_index=True)
        padded_variance = pd.concat(
            [variance, pd.Series(np.full(padding, np.inf))], ignore_index=True
        )
        padded_parent = pd.Series(np.full(len(padded_estimate), 5.0))
        posterior, actual = target_inference.predict_random_effect(
            padded_estimate, padded_variance, padded_parent
        )

        self.assertEqual(actual, expected)
        np.testing.assert_allclose(posterior.iloc[-padding:], 5.0)

    def test_unseen_pitch_type_uses_finite_hierarchical_fallback(self):
        train = sample_frame()
        test = train.iloc[[0]].copy()
        test["pitch_type"] = "SL"

        target_x, target_z = target_inference.infer_targets(train, test)

        self.assertTrue(np.isfinite(target_x).all())
        self.assertTrue(np.isfinite(target_z).all())

    def test_degenerate_pitch_type_plane_stays_finite(self):
        frame = sample_frame()
        rare = frame.iloc[:2].copy()
        rare["pitch_type"] = "KN"
        frame = pd.concat([frame, rare], ignore_index=True)

        target_x, target_z = target_inference.infer_targets(frame, frame)

        self.assertTrue(np.isfinite(target_x).all())
        self.assertTrue(np.isfinite(target_z).all())

    def test_validation_rejects_non_finite_targets(self):
        frame = sample_frame().iloc[:2]

        def broken_method(train, test):
            return np.full(len(test), np.nan), np.full(len(test), np.nan)

        with self.assertRaisesRegex(ValueError, "non-finite target"):
            opencommand.missed(broken_method, frame, frame)


if __name__ == "__main__":
    unittest.main()
