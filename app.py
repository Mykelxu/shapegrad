"""Local ShapeGrad experiment studio."""
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import torch
from shapegrad import fit
from reconstruct import reconstruct
import joblib
from learn_proposals import CoarseRanker
from skimage.metrics import structural_similarity

VECTOR_METHODS = ("greedy", "learned", "coarse")

st.set_page_config(page_title="ShapeGrad Studio", page_icon="S", layout="wide")
torch.set_num_threads(4)
st.title("ShapeGrad Studio")
st.write("Rebuild a picture with overlapping triangles, rectangles, and ellipses. Start with a plain canvas and watch optimized shapes form the image. Compare residual-guided search with learned proposal ranking.")
methods = {"Residual-guided search": "greedy", "Learned tree proposal ranker": "learned", "Coarse heuristic (no ML)": "coarse", "Gradient descent (Adam)": "adam", "Hill climbing": "hill", "Simulated annealing": "anneal"}
with st.sidebar:
    st.header("Experiment")
    upload = st.file_uploader("Your image", type=["png", "jpg", "jpeg", "webp"])
    chosen = st.selectbox("Optimizer", list(methods))
    family = st.selectbox("Primitive family (geometric modes)", ["mixed", "triangle", "rectangle", "ellipse"])
    compare = st.checkbox("Compare search, learned, and heuristic")
    st.subheader("Composition")
    coherent = st.checkbox("Coherent triangle style", value=False, help="For sequential methods: triangles only, consistent half opacity, and no angles below 15 degrees. A style experiment; it does not guarantee lower pixel error.")
    count = st.slider("Shapes", 8, 512, 128, 8)
    st.subheader("Search budget")
    steps = st.slider("Adam / global search iterations", 25, 5000, 600, 25)
    candidates = st.slider("Candidates per search start", 8, 1024, 256, 8)
    refinements = st.slider("Refinement attempts per start", 8, 512, 96, 8)
    restarts = st.number_input("Search starts per shape", min_value=1, max_value=16, value=4)
    size = st.select_slider("Fit resolution (longest side)", options=[48, 64, 96, 128, 192, 256], value=256)
    seed = st.number_input("Random seed", min_value=0, max_value=1000000, value=7)
    edge = st.slider("Edge emphasis", 0.0, 1.0, .2, .05)
    temperature = st.number_input("Annealing starting temperature", min_value=.00001, max_value=.1, value=.001, format="%.5f")
    with st.expander("Gradient appearance", expanded=methods[chosen] == "adam"):
        sharpen = st.checkbox("Sharpen edges during Adam fitting", value=True, help="Begin with smooth edges for movement, then narrow the edge transition. Final exports always use crisp vector geometry.")
        opacity_floor = st.slider("Minimum gradient shape opacity", 0.0, .9, .15, .05, help="Higher values reduce translucent haze but can hide earlier shapes. Applies to Adam, hill climbing, and annealing only.")
    example = st.selectbox("Example image", ["Landscape", "Portrait"], key='example')
    generate = st.button("Generate art", type="primary")
    st.caption("Try triangles for a faceted style. More search starts improve exploration; more shapes add detail.")
    st.caption("Each accepted shape improves the reconstruction. Choose a primitive family and increase the shape or search budget for more detail.")
try:
    example_path = 'examples/portrait-source.png' if example == 'Portrait' else 'examples/target.png'
    source = ImageOps.exif_transpose(Image.open(upload if upload else Path(__file__).parent / example_path)).convert("RGBA")
    source = Image.alpha_composite(Image.new("RGBA", source.size, "white"), source).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()
left, right = st.columns(2)
left.image(source, caption="Source image", width="stretch")
live = right.empty()
live.info("Upload a photo or try the included landscape, then select Generate art.")
st.caption("Residual-guided search starts with a plain canvas and adds one optimized geometric shape at a time. The learned method ranks proposals with an Extra Trees regressor trained on synthetic scenes. Adam uses gradients; geometric optimizers support the chosen primitive family. Hill climbing accepts improvements. Annealing can accept worse proposals early to explore the search space. New search methods minimize pixel MSE; legacy methods use pixel + edge loss. Their loss values are not directly comparable.")
if generate:
    image = source.copy()
    image.thumbnail((size, size))
    target = torch.tensor(np.asarray(image).copy(), dtype=torch.float32) / 255
    progress = st.progress(0)
    status = st.empty()
    results = []
    selected = [(label, methods[label]) for label in ("Residual-guided search", "Learned tree proposal ranker", "Coarse heuristic (no ML)")] if compare else [(chosen, methods[chosen])]
    model_path = Path(__file__).parent / "models/proposal_ranker.joblib"
    if any(method == "learned" for _, method in selected) and not model_path.exists():
        st.error("Train the proposal model first: python learn_proposals.py --benchmark")
        st.stop()
    for run, (label, method) in enumerate(selected):
        frames = []
        started = time.perf_counter()
        def callback(step, loss, rendered):
            array = rendered.clamp(0, 1).cpu().numpy() if isinstance(rendered, torch.Tensor) else rendered.clip(0, 1)
            preview = Image.fromarray((array * 255).astype("uint8"))
            frames.append(preview)
            live.image(preview, caption=f"{label}: iteration {step}", width="stretch")
            progress.progress(min(1., (run + step / (count if method in ("greedy", "learned", "coarse") else steps)) / len(selected)))
            status.write(f"{label} | loss {loss:.6f}")
        if method in ("greedy", "learned", "coarse"):
            model = joblib.load(model_path) if method == "learned" else (CoarseRanker() if method == "coarse" else None)
            effective_family = 'triangle' if coherent else family
            scene, history, evaluations = reconstruct(target.numpy(), count, candidates, refinements, int(seed), model, callback, restarts=int(restarts), shape_family=effective_family, min_triangle_angle=15 if coherent else 0, fixed_opacity=128/255 if coherent else None)
        else:
            scene, history = fit(target, count, steps, int(seed), edge, callback, method, temperature, shape_family=family, sharpen=sharpen, opacity_floor=opacity_floor)
            evaluations = None
        elapsed = time.perf_counter() - started
        with torch.no_grad():
            array = scene.render(*target.shape[:2]) if method in VECTOR_METHODS else scene(*target.shape[:2]).clamp(0, 1).numpy()
            preview = Image.fromarray((array.clip(0, 1) * 255).astype("uint8"))
        # Re-render vector geometry rather than enlarging the fitting bitmap.
        svg = scene.svg(*source.size)
        output_width = round(1024 * source.width / max(source.size))
        output_height = round(1024 * source.height / max(source.size))
        if method in VECTOR_METHODS:
            export_scene = scene
        else:
            export_scene = scene.vector_scene()
        fitted = export_scene.render(*target.shape[:2])
        preview = Image.fromarray((fitted.clip(0, 1) * 255).astype("uint8"))
        live.image(svg, caption="Completed vector reconstruction", width="stretch")
        output_image = Image.fromarray((export_scene.render(max(1, output_height), max(1, output_width)).clip(0, 1)*255).astype('uint8'))
        png = BytesIO()
        output_image.save(png, format="PNG")
        gif = BytesIO()
        frames.append(preview)
        frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:], duration=120, loop=0)
        settings = dict(shapes=count, steps=steps, size=size, seed=int(seed), edge_weight=edge, temperature=temperature, method=method, candidates=candidates, refinements=refinements, restarts=int(restarts), shape_family=effective_family if method in VECTOR_METHODS else family, coherent_style=coherent and method in ('greedy', 'learned', 'coarse'), scoring_backend='regional' if method in ('greedy', 'learned', 'coarse') else 'torch', sharpen=sharpen if method == 'adam' else None, opacity_floor=opacity_floor if method not in VECTOR_METHODS else None)
        output_mse = float(np.mean((fitted - target.numpy()) ** 2))
        window = min(7, min(target.shape[:2]))
        window = window if window % 2 else window - 1
        ssim = float(structural_similarity(target.numpy(), fitted, channel_axis=-1, data_range=1, win_size=window)) if window >= 3 else None
        metrics = dict(output_mse=output_mse, ssim=ssim, svg_bytes=len(svg.encode()), exact_evaluations=evaluations, objective="exported pixel MSE" if method in VECTOR_METHODS else "pixel + edge", settings=settings, initial_loss=history[0], best_loss=min(history), seconds=elapsed, loss_history=history)
        results.append(dict(label=label, method=method, source=image.copy(), residual=np.abs(fitted-target.numpy()), image=preview, png=png.getvalue(), gif=gif.getvalue(), svg=svg, metrics=metrics))
    st.session_state["results"] = results
    progress.empty()
    status.empty()
if "results" in st.session_state:
    results = st.session_state["results"]
    st.subheader("Last completed experiment")
    st.caption("Results keep the settings and source from when Generate art was clicked.")
    st.dataframe([dict(Optimizer=r["label"], Pixel_MSE=r["metrics"]["output_mse"], SSIM=r["metrics"]["ssim"], SVG_KB=round(r["metrics"]["svg_bytes"] / 1024, 1), Seconds=round(r["metrics"]["seconds"], 2)) for r in results], hide_index=True)
    import pandas as pd
    st.line_chart(pd.DataFrame({r["label"]: pd.Series(r["metrics"]["loss_history"]) for r in results}), x_label="Iteration", y_label="Loss")
    st.caption("Geometric search curves count shape additions. Pixel MSE and SSIM in the table use the actual exported geometry at fitting resolution. Learned and heuristic modes consider 4x the candidate pool but use the same number of full-resolution evaluations. Their feature overhead is included in elapsed time. Equal iteration counts are not equal compute budgets. Compare both loss and elapsed time; one image or seed is not a general benchmark.")
    for tab, result in zip(st.tabs([r["label"] for r in results]), results):
        with tab:
            source_col, art_col = st.columns(2)
            source_col.image(result["source"], caption="Source at fitting resolution", width="stretch")
            art_col.image(result["svg"], caption="Crisp vector output", width="stretch")
            m = result["metrics"]
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Pixel error", f'{m["output_mse"]:.5f}')
            metrics_cols[1].metric("Structural similarity", f'{m["ssim"]:.3f}' if m["ssim"] is not None else "N/A")
            metrics_cols[2].metric("Runtime", f'{m["seconds"]:.1f} s')
            with st.expander("Where the reconstruction still differs"):
                st.image(result["residual"].clip(0, 1), caption="Absolute RGB error: brighter areas have larger differences", width="stretch")
            cols = st.columns(4)
            for col, label, key, extension, mime in zip(cols, ["SVG", "PNG", "Animation", "Metrics"], ["svg", "png", "gif", "metrics"], ["svg", "png", "gif", "json"], ["image/svg+xml", "image/png", "image/gif", "application/json"]):
                data = json.dumps(result[key], indent=2) if key == "metrics" else result[key]
                col.download_button(f"Download {label}", data, f"shapegrad-{result['method']}.{extension}", mime, key=f"{result['method']}-{key}")

with st.expander("ML model and benchmark"):
    st.write("The supervised target is the actual reduction in reconstruction MSE from a candidate shape. Twelve geometry and coarse residual features feed an Extra Trees regressor. Training and validation are split by image, not by proposal.")
    card_path = Path(__file__).parent / "models/model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text())
        st.json(card)
    st.caption("The shipped experiment uses synthetic art. Generalization to photographs is unproven; ranking overhead can outweigh savings in exact scoring.")
