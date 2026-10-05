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

st.set_page_config(page_title="ShapeGrad Studio", page_icon="?", layout="wide")
torch.set_num_threads(4)
st.title("ShapeGrad Studio")
st.write("Turn a picture into editable geometric art. Explore how three optimizers learn the same scene.")
methods = {"Gradient descent (Adam)": "adam", "Hill climbing": "hill", "Simulated annealing": "anneal"}
with st.sidebar:
    st.header("Experiment")
    upload = st.file_uploader("Your image", type=["png", "jpg", "jpeg", "webp"])
    chosen = st.selectbox("Optimizer", list(methods))
    compare = st.checkbox("Compare all three methods")
    count = st.slider("Ellipses", 8, 128, 32, 8)
    steps = st.slider("Iterations", 25, 1000, 150, 25)
    size = st.select_slider("Fit resolution (longest side)", options=[48, 64, 96, 128, 192], value=96)
    seed = st.number_input("Random seed", min_value=0, max_value=1000000, value=7)
    edge = st.slider("Edge emphasis", 0.0, 1.0, .2, .05)
    temperature = st.number_input("Annealing starting temperature", min_value=.00001, max_value=.1, value=.001, format="%.5f")
    generate = st.button("Generate art", type="primary")
    st.caption("CPU runs. Start small; more shapes and pixels take longer.")
try:
    source = ImageOps.exif_transpose(Image.open(upload if upload else Path(__file__).parent / "examples/target.png")).convert("RGBA")
    source = Image.alpha_composite(Image.new("RGBA", source.size, "white"), source).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()
left, right = st.columns(2)
left.image(source, caption="Source image", width="stretch")
live = right.empty()
live.info("Upload a photo or try the included landscape, then select Generate art.")
st.caption("Adam uses gradients. Hill climbing accepts improvements. Annealing can accept worse proposals early to explore the search space. All three start from the same seeded scene and minimize the same pixel + edge loss.")
if generate:
    image = source.copy()
    image.thumbnail((size, size))
    target = torch.tensor(np.asarray(image).copy(), dtype=torch.float32) / 255
    progress = st.progress(0)
    status = st.empty()
    results = []
    selected = list(methods.items()) if compare else [(chosen, methods[chosen])]
    for run, (label, method) in enumerate(selected):
        frames = []
        started = time.perf_counter()
        def callback(step, loss, rendered):
            preview = Image.fromarray((rendered.clamp(0, 1).cpu().numpy() * 255).astype("uint8"))
            frames.append(preview)
            live.image(preview, caption=f"{label}: iteration {step}", width="stretch")
            progress.progress((run + step / steps) / len(selected))
            status.write(f"{label} ? loss {loss:.6f}")
        scene, history = fit(target, count, steps, int(seed), edge, callback, method, temperature)
        elapsed = time.perf_counter() - started
        with torch.no_grad():
            preview = Image.fromarray((scene(*target.shape[:2]).clamp(0, 1).numpy() * 255).astype("uint8"))
        png = BytesIO()
        preview.save(png, format="PNG")
        gif = BytesIO()
        frames.append(preview)
        frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:], duration=120, loop=0)
        settings = dict(shapes=count, steps=steps, size=size, seed=int(seed), edge_weight=edge, temperature=temperature, method=method)
        metrics = dict(settings=settings, initial_loss=history[0], best_loss=min(history), seconds=elapsed, loss_history=history)
        results.append(dict(label=label, method=method, image=preview, png=png.getvalue(), gif=gif.getvalue(), svg=scene.svg(*source.size), metrics=metrics))
    st.session_state["results"] = results
    progress.empty()
    status.empty()
if "results" in st.session_state:
    results = st.session_state["results"]
    st.subheader("Last completed experiment")
    st.caption("Results keep the settings and source from when Generate art was clicked.")
    st.dataframe([dict(Optimizer=r["label"], Initial=r["metrics"]["initial_loss"], Best=r["metrics"]["best_loss"], Seconds=round(r["metrics"]["seconds"], 2)) for r in results], hide_index=True)
    st.line_chart({r["label"]: r["metrics"]["loss_history"] for r in results}, x_label="Iteration", y_label="Pixel + edge loss")
    st.caption("Equal iteration counts are not equal compute budgets. Compare both loss and elapsed time; one image or seed is not a general benchmark.")
    for tab, result in zip(st.tabs([r["label"] for r in results]), results):
        with tab:
            st.image(result["image"], caption="Best fitted scene", width=480)
            cols = st.columns(4)
            for col, label, key, extension, mime in zip(cols, ["SVG", "PNG", "Animation", "Metrics"], ["svg", "png", "gif", "metrics"], ["svg", "png", "gif", "json"], ["image/svg+xml", "image/png", "image/gif", "application/json"]):
                data = json.dumps(result[key], indent=2) if key == "metrics" else result[key]
                col.download_button(f"Download {label}", data, f"shapegrad-{result['method']}.{extension}", mime, key=f"{result['method']}-{key}")
