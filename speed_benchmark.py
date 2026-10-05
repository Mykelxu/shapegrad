"""Compare identical search budgets with reference, cached, and regional scoring."""
import json
from pathlib import Path
import time
import joblib
import numpy as np
from learn_proposals import synthetic
from reconstruct import reconstruct


def main():
    rows = []
    model_path = Path('models/proposal_ranker.joblib')
    model = joblib.load(model_path) if model_path.exists() else None
    for image_seed in (2000, 2001, 2002):
        target = synthetic(image_seed, 192)
        for seed in (7, 19):
            reference = None
            for backend in ('reference', 'cached', 'regional'):
                started = time.perf_counter()
                scene, history, evaluations = reconstruct(target, count=16, candidates=48, refinements=40, seed=seed, backend=backend)
                elapsed = time.perf_counter() - started
                pixels = scene.render(192, 192)
                if reference is None:
                    reference = pixels
                row = dict(image_seed=image_seed, seed=seed, backend=backend, mode='search', seconds=elapsed, mse=history[-1], evaluations=evaluations, max_pixel_difference=float(np.abs(pixels-reference).max()), mean_pixel_difference=float(np.abs(pixels-reference).mean()))
                rows.append(row)
                print(json.dumps(row), flush=True)
        if model is not None:
            reference = None
            for backend in ('reference', 'cached', 'regional'):
                started = time.perf_counter()
                scene, history, evaluations = reconstruct(target, count=8, candidates=16, refinements=16, seed=7, model=model, backend=backend)
                elapsed = time.perf_counter() - started
                pixels = scene.render(192, 192)
                if reference is None:
                    reference = pixels
                rows.append(dict(image_seed=image_seed, seed=7, backend=backend, mode='learned', seconds=elapsed, mse=history[-1], evaluations=evaluations, max_pixel_difference=float(np.abs(pixels-reference).max()), mean_pixel_difference=float(np.abs(pixels-reference).mean())))
    output = Path('benchmarks/speed')
    output.mkdir(exist_ok=True, parents=True)
    report = dict(rows=rows, settings=dict(resolution=192, search_shapes=16, candidates=48, refinements=40, learned_shapes=8, learned_candidates=16, learned_refinements=16, restarts=1), note='Same proposal budget and seed. Both optimized scorers retain the dense reduction layout to preserve tie-breaking. Check the recorded pixel differences; cross-platform bitwise equivalence is not promised. Timings exclude final export rendering. Synthetic workload; results are not a Primitive speed comparison.')
    (output / 'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
