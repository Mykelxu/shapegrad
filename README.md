# ShapeGrad

Image-to-SVG reconstruction with residual-guided geometric search and a learned tree model for ranking shape proposals. Includes a local experiment UI, reproducible training pipeline, and held-out benchmark with a non-ML ablation.

## Run locally

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python learn_proposals.py --benchmark
python -m streamlit run app.py
```

Open http://localhost:8501. Upload an image or try the example. **Residual-guided search** is the default. Compare it with **Learned tree proposal ranker** and **Coarse heuristic (no ML)**. Download SVG, PNG, GIF, and metrics. The site runs locally; no hosted deployment is configured.

The locally trained model and dataset are ignored by Git. Training regenerates them in `models/`; the model card is included in the repository. Train before selecting learned mode. The UI loads only the project's local model artifact.

## How reconstruction works

1. Start with the image's mean color.
2. Sample triangle, rectangle, and ellipse candidates near pixels with large residual errors.
3. Solve each candidate's RGB color analytically under alpha compositing, rather than randomly searching color.
4. Refine the strongest candidate with progressively smaller geometric mutations.
5. Recheck the candidate with supersampled antialiasing and commit it only if MSE improves.
6. Repeat, freezing accepted shapes. Raster previews and SVG use the same geometry and layer order, though rasterizer boundary conventions can differ slightly.

The default uses 128 shape additions, 48 candidates, 40 refinement attempts per shape, and 192-pixel fitting resolution. More detailed photographs may need 256 shapes and larger budgets. This remains an approximation; tiny text and fine textures are difficult.

![Improved reconstruction](examples/improved.png)

See [editable output](examples/improved.svg) and [example metrics](examples/improved-metrics.json). The earlier ellipse-only [preview](examples/result.png) remains for comparison. The example is original synthetic artwork, not a photo benchmark.

## The ML component

An `ExtraTreesRegressor` predicts a candidate's full-resolution MSE improvement from twelve features: shape identity, size, coverage, opacity, coarse residual statistics, color variation, and gain measured on a 12?12 proxy image. This is supervised learning of a proposal-value surrogate, not a pretrained generative image model.

The learned method predicts scores for a pool four times larger than the baseline, then fully evaluates only the highest-ranked candidates. Acceptance still requires measured improvement; model predictions alone cannot make the output worse at an individual accepted step.

Training generates 6,100 labeled proposals from 24 synthetic images, over eight intermediate canvas states per image. Validation uses six disjoint image seeds and 1,524 proposals. Entire images are kept separate to avoid proposal-level leakage. Benchmark images use a third disjoint seed range.

The current model achieves **R? = 0.790** and **MAE = 0.000303** on held-out proposal gains. These evaluate gain prediction, not final image aesthetics. The coarse gain feature accounts for about 83% of tree impurity importance; the heuristic ablation is therefore essential.

## Initial benchmark

Three unseen synthetic images ? two search seeds; 32 shape additions, 12 exact candidate evaluations and 12 refinement evaluations per shape, 96-pixel images. Ranking methods get the same full-resolution scoring budget as baseline, but consider four times as many candidates through proxy features.

| Method | Mean final MSE ? | Mean seconds ? |
| --- | ---: | ---: |
| Residual-guided baseline | 0.01568 | 0.92 |
| Coarse gain heuristic, no training | 0.01337 | 2.40 |
| Learned Extra Trees ranking | 0.01296 | 2.65 |

The learned method reduces mean MSE by **17.3% versus baseline**, and **3.1% versus the coarse heuristic**, but takes about **2.9? baseline time**. There is no demonstrated speedup. Timings include feature extraction and inference and depend on the machine. This small synthetic benchmark provides no evidence yet of generalization to photographs or statistical significance.

Raw runs and timing details: [benchmark JSON](benchmarks/results.json). Training setup, split seeds, validation metrics, and feature importance: [model card](models/model_card.json).

## Original gradient experiment

`shapegrad.py` retains the initial differentiable ellipse renderer, Adam optimization, and legacy global hill climbing / simulated annealing for educational comparison:

```sh
python shapegrad.py photo.jpg --method adam --shapes 128 --steps 1000 --size 192 --gif
```

Its pixel-plus-edge objective differs from the new renderer's pixel MSE, so the loss values are not directly comparable. It has softer boundaries and limited shape types and is not the recommended quality mode.

## Validation and next work

```sh
python -m unittest discover -s tests -v
```

Tests check optimal color solving, monotonic accepted error, raster consistency, SVG structure, learned-ranking integration, finite datasets, legacy optimizer behavior, and a full UI generation/export run.

Next experiments: train and evaluate on a licensed photo corpus, cache coarse features, compare under equal wall-clock budgets, add confidence intervals across more images/seeds, and test neural proposal prediction. Perceptual features or semantic importance masks could prioritize subjects, but are not implemented.

A defensible resume description: ?Built an image-to-vector reconstruction system with analytical color fitting, residual-guided search, and an Extra Trees proposal-ranking surrogate; evaluated against search and non-ML heuristic baselines using image-disjoint splits.? Include the measured quality/runtime tradeoff rather than claiming generative-model training or acceleration.

## Inspiration and license

[Primitive](https://github.com/fogleman/primitive) inspired sequential shape search. [DiffVG](https://github.com/BachiLi/diffvg) provides related differentiable-vector research. This repository contains an original implementation; neither project's code is bundled. The underlying techniques are established, not claimed as novel research.

MIT. Synthetic examples are generated locally using Pillow.
