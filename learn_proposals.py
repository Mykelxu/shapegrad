"""Train a tree surrogate and benchmark on disjoint synthetic image seeds."""
import argparse
import json
from pathlib import Path
import time
import joblib
import sklearn
import platform
import numpy as np
from PIL import Image, ImageDraw
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from reconstruct import proposal, score, features, reconstruct, FEATURE_NAMES


def synthetic(seed, size=64):
    rng = np.random.default_rng(seed)
    image = Image.new("RGB", (size, size), tuple(rng.integers(0, 256, 3)))
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        coords = rng.integers(0, size, (3, 2))
        color = tuple(rng.integers(0, 256, 3))
        kind = rng.choice(["ellipse", "rectangle", "polygon"])
        if kind == "polygon":
            draw.polygon([tuple(p) for p in coords], fill=color)
        else:
            lo, hi = coords[:2].min(0), coords[:2].max(0)
            getattr(draw, kind)(tuple(np.concatenate((lo, hi))), fill=color)
    return np.asarray(image, dtype=np.float32) / 255


def collect(seeds, stages=8, proposals=32):
    x, y, groups = [], [], []
    for seed in seeds:
        target = synthetic(seed)
        canvas = np.broadcast_to(target.mean((0, 1)), target.shape).copy()
        rng = np.random.default_rng(seed + 10000)
        for _ in range(stages):
            best = None
            for _ in range(proposals):
                p = proposal(target, canvas, rng)
                gain, color, alpha = score(p, target, canvas)
                if not np.isfinite(gain):
                    continue
                x.append(features(p, target, canvas)); y.append(gain); groups.append(seed)
                if best is None or gain > best[0]:
                    best = (gain, color, alpha)
            if best and best[0] > 0:
                _, color, alpha = best
                canvas = canvas * (1 - alpha[..., None]) + alpha[..., None] * color
    return np.asarray(x), np.asarray(y), np.asarray(groups)


def train(output=Path("models"), images=24):
    output.mkdir(exist_ok=True, parents=True)
    train_seeds = list(range(images))
    test_seeds = list(range(1000, 1006))
    x, y, groups = collect(train_seeds)
    xt, yt, test_groups = collect(test_seeds)
    model = ExtraTreesRegressor(n_estimators=80, max_depth=14, min_samples_leaf=3, random_state=7, n_jobs=1)
    model.fit(x, y)
    prediction = model.predict(xt)
    report = dict(python_version=platform.python_version(), sklearn_version=sklearn.__version__, numpy_version=np.__version__, training_settings=dict(stages=8, proposals_per_stage=32, trees=80, max_depth=14, min_samples_leaf=3, random_seed=7), train_image_seeds=train_seeds, test_image_seeds=test_seeds,
        train_samples=len(y), test_samples=len(yt), held_out_r2=r2_score(yt, prediction),
        held_out_mae=mean_absolute_error(yt, prediction),
        feature_importance=dict(zip(FEATURE_NAMES, model.feature_importances_.tolist())),
        limitations="Synthetic images only. Photo generalization and runtime improvement are unproven.")
    joblib.dump(model, output / "proposal_ranker.joblib")
    np.savez_compressed(output / "training_data.npz", x=x, y=y, image_seed=groups, test_x=xt, test_y=yt, test_image_seed=test_groups)
    (output / "model_card.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return model


class CoarseRanker:
    """Ablation: rank directly by coarse gain without a trained model."""
    def predict(self, x):
        return np.asarray(x)[:, -1]


def benchmark(model, output=Path("benchmarks")):
    output.mkdir(exist_ok=True, parents=True)
    rows = []
    for image_seed in range(2000, 2003):
        target = synthetic(image_seed, 96)
        for seed in [7, 19]:
            for name, ranker in [("baseline", None), ("coarse_heuristic", CoarseRanker()), ("learned", model)]:
                start = time.perf_counter()
                scene, losses, evaluations = reconstruct(target, count=32, candidates=12, refinements=12, seed=seed, model=ranker)
                rows.append(dict(image_seed=image_seed, seed=seed, method=name, mse=losses[-1], psnr_db=float(-10 * np.log10(max(losses[-1], 1e-12))), seconds=time.perf_counter()-start, exact_evaluations=evaluations))
                Image.fromarray((scene.render(96, 96).clip(0, 1)*255).astype("uint8")).save(output / f"{image_seed}-{seed}-{name}.png")
        print(f"Benchmarked image {image_seed}", flush=True)
    report = dict(rows=rows, note="Same full-resolution scoring budget; learned mode ranks 4x candidates with low-resolution features. Feature extraction and inference are included in wall time. This is a small synthetic benchmark, not evidence of photo generalization.")
    (output / "results.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--images", type=int, default=24)
    args = parser.parse_args()
    model = train(images=args.images)
    if args.benchmark:
        benchmark(model)
