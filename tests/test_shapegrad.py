import unittest
import xml.etree.ElementTree as ET
import torch
from shapegrad import ShapeScene, fit


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

    def test_seed(self):
        target = torch.rand(8, 8, 3)
        self.assertTrue(torch.equal(ShapeScene(target, 3)(8, 8), ShapeScene(target, 3)(8, 8)))


if __name__ == '__main__':
    unittest.main()
