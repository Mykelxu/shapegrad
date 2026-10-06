"""Compare contour initialization/refinement and random search on art and a photo."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from skimage import data
from skimage.metrics import structural_similarity
import torch
from contour_model import contour_fit
from reconstruct import reconstruct


def main():
    torch.set_num_threads(4)
    output = Path('benchmarks/contours')
    output.mkdir(exist_ok=True, parents=True)
    inputs = [('landscape', Image.open('examples/target.png').convert('RGB')),
              ('astronaut', Image.fromarray(data.astronaut()))]
    rows = []
    for name, source in inputs:
        source.thumbnail((128, 128))
        target = np.asarray(source, dtype=np.float32) / 255
        source.save(output / (name + '-source.png'))
        for method in ('search', 'contour_init', 'contour_refined'):
            started = time.perf_counter()
            if method == 'search':
                scene, history, evaluations = reconstruct(target, count=128, candidates=48, refinements=40)
                details = dict(evaluations=evaluations, shapes=len(scene.shapes),
                    vertices=sum(3 if s.kind == 'triangle' else 4 if s.kind == 'rectangle' else 0 for s in scene.shapes))
            else:
                scene, history, details = contour_fit(target, count=128, palette=24,
                    steps=40 if method == 'contour_refined' else 0)
                details['pipeline'] = details.pop('method')
            elapsed = time.perf_counter() - started
            rendered = scene.render(*target.shape[:2])
            mse = float(np.mean((rendered - target) ** 2))
            row = dict(image=name, method=method, seconds=elapsed, mse=mse,
                ssim=float(structural_similarity(target, rendered, channel_axis=-1, data_range=1)),
                svg_bytes=len(scene.svg(source.width, source.height).encode()), **details)
            rows.append(row)
            Image.fromarray((scene.render(source.height*3, source.width*3).clip(0, 1)*255).astype('uint8')).save(output / (name + '-' + method + '.png'))
            (output / (name + '-' + method + '.svg')).write_text(scene.svg(source.width, source.height))
            print(json.dumps(row), flush=True)
    report = dict(rows=rows, settings=dict(fitting_resolution=128, region_or_shape_budget=128, palette_budget=24, refinement_steps=40, search_candidates=48, search_refinements=40, search_starts=1, seed=7),
        note='Two examples, not a general benchmark. Contour paths permit arbitrary vertices, unlike geometric primitives; budgets and representational complexity differ. Compare SVG bytes and vertex counts as well as quality and time. Refinement only accepts exported scenes with lower MSE.',
        photo_attribution='NASA photograph of Eileen Collins, public domain, provided by skimage.data.astronaut.',
        photo_source='https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut')
    (output / 'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
