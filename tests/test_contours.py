import unittest
import xml.etree.ElementTree as ET
import numpy as np
import torch
from contour_model import Region, ContourScene, region_mask, initialize_regions, SharedRefinement, contour_fit, structural_loss
from skimage.metrics import structural_similarity


class ContourTests(unittest.TestCase):
    def test_differentiable_structural_loss(self):
        rng = np.random.default_rng(7)
        target = rng.random((24,32,3), dtype=np.float32)
        predicted = (target*.8 + .1).astype(np.float32)
        tensor = torch.tensor(predicted, requires_grad=True)
        loss = structural_loss(tensor, torch.from_numpy(target))
        expected = 1 - structural_similarity(target, predicted, channel_axis=-1, data_range=1)
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_holes_and_svg(self):
        outer = np.array([[.1,.1],[.9,.1],[.9,.9],[.1,.9],[.1,.1]], dtype=np.float32)
        inner = np.array([[.4,.4],[.6,.4],[.6,.6],[.4,.6],[.4,.4]], dtype=np.float32)
        region = Region([outer, inner], np.array([1.,0.,0.], dtype=np.float32))
        coverage = region_mask(region, 64, 64)
        self.assertLess(coverage[32,32], .01)
        self.assertGreater(coverage[16,16], .99)
        root = ET.fromstring(ContourScene(np.zeros(3), [region]).svg(640,640))
        path = root.find('{http://www.w3.org/2000/svg}path')
        self.assertEqual(path.attrib['fill-rule'], 'evenodd')
        self.assertEqual(path.attrib['d'].count('M'), 2)

    def test_regions_gradients_and_export_acceptance(self):
        torch.set_num_threads(2)
        target = np.ones((32,48,3), dtype=np.float32) * .8
        target[8:24,12:36] = [.9,.1,.2]
        initial, details = initialize_regions(target, count=8, palette=4)
        self.assertLess(len(initial.regions), 8)
        model = SharedRefinement(initial, 32, 48)
        model().square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        scene, history, metadata = contour_fit(target, count=8, palette=4, steps=10)
        mse = float(np.mean((scene.render(32,48)-target)**2))
        self.assertAlmostEqual(mse, metadata['best_mse'], places=7)
        self.assertLessEqual(mse, metadata['initialized_mse'])
        self.assertGreaterEqual(metadata['best_ssim'], metadata['initialized_ssim'])
        self.assertLess(mse, history[0])
        self.assertTrue(all(b <= a for a,b in zip(history[1:], history[2:])))
        ET.fromstring(scene.svg(480,320))

    def test_shared_boundary_mapping(self):
        shared = np.array([.5,.5], dtype=np.float32)
        first = Region([np.array([[.1,.1], shared, [.1,.9], [.1,.1]], dtype=np.float32)], np.array([.8,.1,.2], dtype=np.float32))
        second = Region([np.array([[.9,.1], [.9,.9], shared, [.9,.1]], dtype=np.float32)], np.array([.1,.8,.2], dtype=np.float32))
        model = SharedRefinement(ContourScene(np.zeros(3), [first, second]), 32,48)
        with torch.no_grad():
            model.displacement.fill_(.2)
        exported = model.export_scene()
        np.testing.assert_array_equal(exported.regions[0].contours[0][1], exported.regions[1].contours[0][2])

    def test_uniform_and_seeded_inputs(self):
        target = np.ones((8,8,3), dtype=np.float32)*.4
        scene, history, metadata = contour_fit(target, steps=10)
        self.assertEqual(len(scene.regions), 0)
        np.testing.assert_allclose(scene.render(8,8), target, atol=1e-6)
        target[2:6,2:6] = [.8,.1,.2]
        a, _, _ = contour_fit(target, count=4, palette=2, steps=0)
        b, _, _ = contour_fit(target, count=4, palette=2, steps=0)
        self.assertEqual(a.svg(8,8), b.svg(8,8))
