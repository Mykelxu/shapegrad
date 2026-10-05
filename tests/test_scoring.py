import unittest
import numpy as np
from reconstruct import ScoringContext, proposal, score, reconstruct, features
from learn_proposals import synthetic


class ScoringTests(unittest.TestCase):
    def test_candidate_equivalence(self):
        target, canvas = synthetic(41), synthetic(42)
        context = ScoringContext(target, canvas)
        rng = np.random.default_rng(4)
        for _ in range(100):
            shape = proposal(target, canvas, rng)
            dense_gain, dense_color, _ = score(shape, target, canvas)
            gain, color, _ = context.evaluate(shape)
            self.assertEqual(gain, dense_gain)
            np.testing.assert_array_equal(color, dense_color)
            np.testing.assert_array_equal(features(shape, target, canvas), features(shape, target, canvas, context.coarse))

    def test_cached_search_identical(self):
        target = synthetic(2000, 48)
        a, ha, ea = reconstruct(target, 8, 8, 8, backend='reference')
        b, hb, eb = reconstruct(target, 8, 8, 8, backend='cached')
        self.assertEqual(ha, hb)
        self.assertEqual(ea, eb)
        self.assertEqual(a.svg(48, 48), b.svg(48, 48))

    def test_region_reconstruction_accuracy(self):
        target = synthetic(2001, 64)
        a, ha, ea = reconstruct(target, 16, 16, 16, backend='reference')
        b, hb, eb = reconstruct(target, 16, 16, 16, backend='regional')
        self.assertEqual(ea, eb)
        self.assertEqual(ha, hb)
        self.assertEqual(a.svg(64, 64), b.svg(64, 64))
        np.testing.assert_array_equal(a.render(64, 64), b.render(64, 64))
