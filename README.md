# ShapeGrad

Repaint images with shapes that learn. A compact PyTorch experiment that fits layered, translucent ellipses to an image and exports editable SVG art.

Instead of searching for one new shape at a time, ShapeGrad jointly optimizes position, radii, color, and opacity with Adam. A soft ellipse renderer makes these parameters differentiable; pixel and edge losses guide the fit. No API key, model download, or training dataset required.

This is per-image gradient-based optimization and is an original educational implementation of an established idea.

## Quick start

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python shapegrad.py examples/target.png --shapes 32 --steps 300 --gif
```

Open `output/result.svg` in a browser or vector editor. `result.png` is the soft raster preview, `progress.gif` shows optimization, and `metrics.json` records losses and settings. SVG keeps the input's aspect ratio and original dimensions. Raster previews use the fitting resolution.

```sh
python shapegrad.py photo.jpg --shapes 128 --steps 1000 --size 192 --output output/photo
python shapegrad.py photo.jpg --device cuda --gif
python -m unittest discover -s tests -v
```

Start with 32–64 shapes and a 128-pixel fitting size on CPU. Cost grows with shape count, pixel count, and iterations. CUDA requires a compatible PyTorch installation. A fixed seed makes CPU runs repeatable in the same environment; cross-device bitwise reproducibility is not promised.

## Example

Original synthetic artwork and a 32-ellipse fit:

![Target](examples/target.png)
![Fitted image](examples/result.png)

See [editable SVG](examples/result.svg) and [optimization animation](examples/progress.gif).

## Limits and next experiments

Only axis-aligned ellipses are implemented. Hard SVG boundaries differ slightly from the soft optimization preview. Small details, text, and sharp corners are difficult; local minima remain possible. Image metadata is not preserved; transparent inputs are composited over white.

- Add triangles, rotated rectangles, and Bezier strokes.
- Add user-supplied importance masks to spend detail on subjects.
- Compare pixel loss with pretrained perceptual features.
- Train a shape-prediction network using fitted scenes as supervision.
- Add temporal consistency for video and a browser playground.

These are planned experiments, not implemented features.

## Inspiration

- [Primitive](https://github.com/fogleman/primitive): shape-based image approximation through search.
- [DiffVG](https://github.com/BachiLi/diffvg): differentiable vector graphics research and tooling.

No code from either project is bundled. Example artwork is generated locally using Pillow.

## License

MIT. Contributions and benchmark comparisons welcome.
