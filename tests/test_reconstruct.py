import unittest
import xml.etree.ElementTree as ET
import numpy as np
from reconstruct import Primitive, score, reconstruct, features, FEATURE_NAMES, triangle_valid
from learn_proposals import collect, CoarseRanker
from sklearn.ensemble import ExtraTreesRegressor


class ReconstructionTests(unittest.TestCase):
    def test_coherent_triangle_style(self):
        target = np.ones((32, 48, 3), dtype=np.float32) * .9
        target[8:24, 12:36] = [.9, .1, .2]
        scene, losses, _ = reconstruct(target, 8, 16, 16, shape_family='triangle', min_triangle_angle=15, fixed_opacity=128/255)
        self.assertLess(losses[-1], losses[0])
        for shape in scene.shapes:
            self.assertEqual(shape.kind, 'triangle')
            self.assertEqual(shape.alpha, 128/255)
            self.assertTrue(triangle_valid(shape.points, 15))

    def test_optimal_color(self):
        target = np.ones((16, 20, 3), dtype=np.float32) * [.7, .2, .4]
        canvas = np.zeros_like(target)
        shape = Primitive("rectangle", np.array([[0, 0], [1, 1]]), 1)
        gain, color, _ = score(shape, target, canvas)
        np.testing.assert_allclose(color, [.7, .2, .4], atol=1e-6)
        self.assertGreater(gain, 0)
        self.assertEqual(len(features(shape, target, canvas)), len(FEATURE_NAMES))

    def test_quality_and_consistent_rendering(self):
        target = np.ones((32, 48, 3), dtype=np.float32) * .9
        target[8:24, 12:36] = [.9, .1, .2]
        scene, losses, evaluations = reconstruct(target, 16, 16, 16)
        self.assertLess(losses[-1], losses[0] * .35)
        self.assertTrue(all(b <= a for a, b in zip(losses, losses[1:])))
        self.assertEqual(evaluations, 16 * 33)
        self.assertAlmostEqual(float(np.mean((scene.render(32, 48) - target) ** 2)), losses[-1], places=6)
        root = ET.fromstring(scene.svg(480, 320))
        self.assertEqual(len(root) - 1, len(scene.shapes))
        ranked, _, _ = reconstruct(target, 4, 8, 8, model=CoarseRanker())
        self.assertTrue(np.isfinite(ranked.render(32, 48)).all())

    def test_dataset_groups_and_labels(self):
        x, y, groups = collect([10, 11], stages=2, proposals=4)
        self.assertEqual(x.shape[1], len(FEATURE_NAMES))
        self.assertEqual(set(groups), {10, 11})
        self.assertTrue(np.isfinite(x).all())
        self.assertTrue(np.isfinite(y).all())
        model = ExtraTreesRegressor(n_estimators=4, min_samples_leaf=2, random_state=7).fit(x, y)
        target = np.ones((16, 16, 3), dtype=np.float32) * .8
        target[4:12, 4:12] = [.9, .1, .2]
        _, losses, _ = reconstruct(target, 4, 8, 8, model=model)
        self.assertLess(losses[-1], losses[0])
