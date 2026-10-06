import unittest
import xml.etree.ElementTree as ET
import torch
from shapegrad import ShapeScene, fit, accept_proposal, objective


class ShapeGradTests(unittest.TestCase):
    def test_optimization_and_svg(self):
        torch.set_num_threads(2)
        target = torch.ones(24, 32, 3) * .9
        target[6:18, 8:24] = torch.tensor([.9, .1, .2])
        scene, losses = fit(target, count=8, steps=60, shape_family="ellipse")
        self.assertLess(min(losses), losses[0] * .75)
        root = ET.fromstring(scene.svg(320, 240))
        self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}ellipse')), 8)
        self.assertEqual(root.attrib['viewBox'], '0 0 320 240')
        rendered = scene(24, 32)
        self.assertTrue(torch.isfinite(rendered).all())
        rendered.mean().backward()
        for parameter in scene.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_search_methods(self):
        target = torch.ones(12, 16, 3) * .8
        target[3:9, 4:12] = torch.tensor([.8, .1, .2])
        for method in ('hill', 'anneal'):
            scene, history = fit(target, count=4, steps=50, method=method)
            self.assertLess(min(history), history[0])
            self.assertEqual(len(history), 51)
            self.assertAlmostEqual(float(objective(scene(12, 16), target).detach()), min(history), places=6)
            if method == 'hill':
                self.assertTrue(all(b <= a for a, b in zip(history, history[1:])))
        self.assertTrue(accept_proposal(-1, 0, .9))
        self.assertFalse(accept_proposal(.1, 0, 0))
        self.assertTrue(accept_proposal(.001, .01, .1))
        self.assertFalse(accept_proposal(.1, .001, .5))

    def test_mixed_shapes_in_every_optimizer(self):
        target = torch.ones(16, 20, 3) * .8
        target[4:12, 5:15] = torch.tensor([.9, .1, .2])
        for method in ('adam', 'hill', 'anneal'):
            scene, losses = fit(target, count=6, steps=20, method=method, shape_family='mixed')
            root = ET.fromstring(scene.svg(200, 160))
            self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}ellipse')), 2)
            self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}polygon')), 2)
            self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}rect')), 3)
            self.assertTrue(torch.isfinite(scene(16, 20)).all())
            self.assertLessEqual(min(losses), losses[0])
        scene = ShapeScene(target, 6)
        scene(16, 20).mean().backward()
        self.assertIsNotNone(scene.vertices.grad)
        self.assertGreater(float(scene.vertices.grad.abs().sum()), 0)

    def test_sharp_adam_and_opacity_export(self):
        target = torch.ones(16, 20, 3) * .8
        target[4:12, 5:15] = torch.tensor([.9, .1, .2])
        scene, history = fit(target, count=6, steps=30, shape_family='triangle', opacity_floor=.6)
        self.assertEqual(scene.sharpness, 160.)
        self.assertAlmostEqual(float(objective(scene(16, 20), target).detach()), min(history), places=6)
        self.assertTrue(bool((scene.alphas() >= .6).all()))
        vector = scene.vector_scene()
        self.assertEqual(len(vector.shapes), 6)
        root = ET.fromstring(scene.svg(200, 160))
        polygons = root.findall('{http://www.w3.org/2000/svg}polygon')
        for polygon, primitive, alpha in zip(polygons, vector.shapes, scene.alphas().detach()):
            self.assertAlmostEqual(float(polygon.attrib['fill-opacity']), float(alpha), places=4)
            self.assertAlmostEqual(primitive.alpha, float(alpha), places=6)
        self.assertTrue(torch.isfinite(torch.tensor(vector.render(16, 20))).all())

    def test_seed(self):
        target = torch.rand(8, 8, 3)
        self.assertTrue(torch.equal(ShapeScene(target, 3)(8, 8), ShapeScene(target, 3)(8, 8)))


if __name__ == '__main__':
    unittest.main()
