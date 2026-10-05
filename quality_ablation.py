"""Measure candidate budget, restarts, and shape count on the bundled target."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from reconstruct import reconstruct


def main():
    image = Image.open('examples/target.png').convert('RGB')
    image.thumbnail((128, 128))
    target = np.asarray(image, dtype=np.float32) / 255
    rows = []
    settings = [dict(name='small_search', count=64, candidates=48, refinements=40, restarts=1),
                dict(name='deeper_search', count=64, candidates=128, refinements=96, restarts=3),
                dict(name='more_shapes', count=128, candidates=128, refinements=96, restarts=3)]
    output = Path('benchmarks/quality')
    output.mkdir(exist_ok=True, parents=True)
    for config in settings:
        args = {k: v for k, v in config.items() if k != 'name'}
        start = time.perf_counter()
        scene, history, evaluations = reconstruct(target, **args)
        seconds = time.perf_counter() - start
        (output / (config['name'] + '.svg')).write_text(scene.svg(1024, 768))
        Image.fromarray((scene.render(384, 512).clip(0, 1)*255).astype('uint8')).save(output / (config['name'] + '.png'))
        row = dict(**config, seed=7, fitting_size=128, final_mse=history[-1], seconds=seconds, evaluations=evaluations)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (output / 'results.json').write_text(json.dumps(dict(rows=rows, note='One synthetic image and one seed; isolates budget effects, not a general benchmark or comparison with Primitive.'), indent=2))


if __name__ == '__main__':
    main()
