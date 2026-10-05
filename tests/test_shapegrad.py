import unittest
import xml.etree.ElementTree as ET
import torch
from shapegrad import ShapeScene, fit, accept_proposal, objective


class ShapeGradTests(unittest.TestCase):
    def test_optimization_and_svg(self):
        torch.set_num_threads(2)
        target = torch.ones(24, 32, 3) * .9
        target[6:18, 8:24] = torch.tensor([.9, .1, .2])
        scene, losses = fit(target, count=8, steps=60)
        self.assertLess(min(losses), losses[0] * .75)
        root = ET.fromstring(scene.svg(320, 240))
        self.assertEqual(len(root.findall('{http://www.w3.org/2000/svg}ellipse')), 8)
        self.assertEqual(root.attrib['viewBox'], '0 0 320 240')
        rendered = scene(24, 32)
        self.assertTrue(torch.isfinite(rendered).all())
        rendered.mean().backward()
        for parameter in scene.parameters():
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

    def test_seed(self):
        target = torch.rand(8, 8, 3)
        self.assertTrue(torch.equal(ShapeScene(target, 3)(8, 8), ShapeScene(target, 3)(8, 8)))


if __name__ == '__main__':
    unittest.main()
