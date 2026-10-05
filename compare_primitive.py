"""Run the installed original Primitive beside ShapeGrad on the bundled target."""
import json
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from PIL import Image
from reconstruct import reconstruct


def main():
    gopath = subprocess.check_output(['go', 'env', 'GOPATH'], text=True).strip()
    executable = Path(gopath.split(';')[0]) / 'bin' / 'primitive.exe'
    if not executable.exists():
        raise SystemExit('First run: go install github.com/fogleman/primitive@latest')
    output = Path('benchmarks/primitive')
    output.mkdir(exist_ok=True, parents=True)
    source = Path('examples/target.png').resolve()
    target = Image.open(source).convert('RGB')
    fitting = target.copy()
    fitting.thumbnail((128, 128))
    array = np.asarray(fitting, dtype=np.float32) / 255
    rows = []
    for label, mode in [('primitive_triangles', '1'), ('primitive_mixed', '0')]:
        command = [str(executable), '-i', str(source), '-o', str((output / (label + '.png')).resolve()), '-o', str((output / (label + '.svg')).resolve()), '-n', '64', '-m', mode, '-r', '128', '-s', '256', '-j', '2', '-v']
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        elapsed = time.perf_counter() - started
        log = result.stdout + result.stderr
        (output / (label + '.log')).write_text(log)
        image = np.asarray(Image.open(output / (label + '.png')).convert('RGB'), dtype=np.float32) / 255
        full_target = np.asarray(target, dtype=np.float32) / 255
        counts = [int(n) for n in re.findall(r'\bn=(\d+)', log)]
        row = dict(method=label, command=command, seconds=elapsed, output_mse=float(np.mean((image - full_target) ** 2)), candidate_evaluations=sum(counts), workers=2, opacity=128/255, shape_additions=64)
        rows.append(row)
        print(json.dumps(row), flush=True)
    for family in ('mixed', 'triangle'):
        started = time.perf_counter()
        scene, history, evaluations = reconstruct(array, count=64, candidates=128, refinements=96, restarts=3, shape_family=family)
        elapsed = time.perf_counter() - started
        image = scene.render(target.height, target.width)
        label = 'shapegrad_' + family
        Image.fromarray((image.clip(0, 1)*255).astype('uint8')).save(output / (label + '.png'))
        (output / (label + '.svg')).write_text(scene.svg(target.width, target.height))
        row = dict(method=label, seconds=elapsed, output_mse=float(np.mean((image - full_target) ** 2)), fitting_mse=history[-1], candidate_evaluations=evaluations, workers=1, opacity='variable', shape_additions=64)
        rows.append(row)
        print(json.dumps(row), flush=True)
    metadata = subprocess.check_output(['go', 'version', '-m', str(executable)], text=True)
    (output / 'upstream-build.txt').write_text(metadata)
    report = dict(rows=rows, fitting_longest_side=128, output_longest_side=256, note='Single synthetic target. Same nominal fitting/output dimensions and shape count, different scoring budgets and rasterizers. Primitive is unseeded and uses fixed opacity; ShapeGrad seed=7 uses variable opacity. Not equal compute or a general benchmark.', upstream='https://github.com/fogleman/primitive', build=metadata)
    (output / 'results.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
