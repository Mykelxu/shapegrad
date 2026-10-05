"""Residual-guided geometric reconstruction with optional learned proposal ranking."""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw

KINDS = ("ellipse", "rectangle", "triangle")
FEATURE_NAMES = ["ellipse", "rectangle", "triangle", "width", "height", "area", "opacity", "residual_mean", "residual_max", "target_std", "canvas_std", "local_gain"]

@dataclass
class Primitive:
    kind: str
    points: np.ndarray
    alpha: float
    color: object = None


def mask_image(shape, height, width, scale=1):
    image = Image.new("L", (width * scale, height * scale))
    draw = ImageDraw.Draw(image)
    points = shape.points * np.array([width * scale, height * scale])
    if shape.kind == "triangle":
        draw.polygon([tuple(p) for p in points], fill=255)
    else:
        box = tuple(np.concatenate((points.min(0), points.max(0))))
        getattr(draw, shape.kind)(box, fill=255)
    if scale > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def mask(shape, height, width, scale=1):
    return np.asarray(mask_image(shape, height, width, scale), dtype=np.float32) / 255


def triangle_valid(points, min_angle):
    if min_angle <= 0:
        return True
    for index in range(3):
        a = points[(index + 1) % 3] - points[index]
        b = points[(index + 2) % 3] - points[index]
        length = np.linalg.norm(a) * np.linalg.norm(b)
        if length < 1e-10:
            return False
        angle = np.degrees(np.arccos(np.clip(np.dot(a, b) / length, -1, 1)))
        if angle < min_angle:
            return False
    return True


def proposal(target, canvas, rng, shape_family="mixed", probabilities=None, min_triangle_angle=0, fixed_opacity=None):
    h, w = target.shape[:2]
    if probabilities is None:
        residual = ((target - canvas) ** 2).mean(-1)
        weights = residual.ravel() + 1e-5
        probabilities = weights / weights.sum()
    index = rng.choice(h * w, p=probabilities)
    center = np.array([(index % w + .5) / w, (index // w + .5) / h])
    radius = np.exp(rng.uniform(np.log(.015), np.log(.6), 2))
    kind = rng.choice(KINDS) if shape_family == "mixed" else shape_family
    if kind == "triangle":
        for attempt in range(64):
            angles = rng.uniform(0, 2 * np.pi, 3)
            points = (center + np.stack([np.cos(angles), np.sin(angles)], -1) * radius).clip(-.25, 1.25)
            if triangle_valid(points, min_triangle_angle):
                break
        else:
            angles = np.arange(3) * 2 * np.pi / 3
            points = center + np.stack([np.cos(angles), np.sin(angles)], -1) * min(radius.min(), .2)
    else:
        points = np.stack([center - radius, center + radius])
    opacity = float(rng.choice([.35, .6, .85, 1.]))
    return Primitive(kind, points.clip(-.25, 1.25), opacity if fixed_opacity is None else fixed_opacity)


def mutate(shape, rng, scale, min_triangle_angle=0, fixed_opacity=None):
    points = shape.points.copy()
    alpha = shape.alpha
    if rng.random() < .85:
        row = rng.integers(len(points))
        axis = rng.integers(2)
        points[row, axis] += rng.normal(0, scale)
    else:
        alpha = float(np.clip(alpha + rng.normal(0, .08), .15, 1))
    points = points.clip(-.25, 1.25)
    if shape.kind == 'triangle' and not triangle_valid(points, min_triangle_angle):
        points = shape.points.copy()
    return Primitive(shape.kind, points, alpha if fixed_opacity is None else fixed_opacity)


def score(shape, target, canvas, antialias=1, base_loss=None):
    alpha = mask(shape, *target.shape[:2], scale=antialias) * shape.alpha
    denominator = np.square(alpha).sum()
    if denominator < 1e-8:
        return -np.inf, np.zeros(3), alpha
    # Least-squares optimal color under alpha compositing.
    color = (alpha[..., None] * (target - (1 - alpha[..., None]) * canvas)).sum((0, 1)) / denominator
    color = color.clip(0, 1)
    new = (1 - alpha[..., None]) * canvas + alpha[..., None] * color
    before = np.mean((target - canvas) ** 2) if base_loss is None else base_loss
    gain = float(before - np.mean((target - new) ** 2))
    return gain, color, alpha


def coarse_images(target, canvas):
    low = lambda a: np.asarray(Image.fromarray((a.clip(0, 1) * 255).astype("uint8")).resize((12, 12), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return low(target), low(canvas)


def features(shape, target, canvas, coarse=None):
    t, c = coarse_images(target, canvas) if coarse is None else coarse
    inside = mask(shape, 12, 12) > 0
    residual = ((t - c) ** 2).mean(-1)
    extent = shape.points.max(0) - shape.points.min(0)
    gain, _, _ = score(shape, t, c)
    return [float(shape.kind == k) for k in KINDS] + [*extent, float(inside.mean()), shape.alpha,
        float(residual[inside].mean()) if inside.any() else 0,
        float(residual[inside].max()) if inside.any() else 0,
        float(t[inside].std()) if inside.any() else 0,
        float(c[inside].std()) if inside.any() else 0,
        gain if np.isfinite(gain) else 0]


class ScoringContext:
    """Cache one fixed canvas state; region scoring uses identical raster coverage."""
    def __init__(self, target, canvas, backend='regional'):
        self.target, self.canvas, self.backend = target, canvas, backend
        self.errors = (target - canvas) ** 2
        self.base_loss = self.errors.mean()
        weights = self.errors.mean(-1).ravel() + 1e-5
        self.probabilities = weights / weights.sum()
        self.coarse = coarse_images(target, canvas)
        # Retain the original reduction layout to preserve tie-breaking exactly.
        self.numerator = np.zeros_like(target)
        self.alpha_squared = np.zeros(target.shape[:2], dtype=np.float32)
        self.candidate_errors = self.errors.copy()

    def evaluate(self, shape):
        if self.backend == 'reference':
            return score(shape, self.target, self.canvas)
        if self.backend == 'cached':
            return score(shape, self.target, self.canvas, base_loss=self.base_loss)
        image = mask_image(shape, *self.target.shape[:2])
        box = image.getbbox()
        if box is None:
            return -np.inf, np.zeros(3), None
        x0, y0, x1, y1 = box
        alpha = np.asarray(image.crop(box), dtype=np.float32) / 255 * shape.alpha
        t, c = self.target[y0:y1, x0:x1], self.canvas[y0:y1, x0:x1]
        self.alpha_squared[y0:y1, x0:x1] = np.square(alpha)
        denominator = self.alpha_squared.sum()
        self.alpha_squared[y0:y1, x0:x1] = 0
        if denominator < 1e-8:
            return -np.inf, np.zeros(3), None
        self.numerator[y0:y1, x0:x1] = alpha[..., None] * (t - (1 - alpha[..., None]) * c)
        color = self.numerator.sum((0, 1)) / denominator
        self.numerator[y0:y1, x0:x1] = 0
        color = color.clip(0, 1)
        new = (1 - alpha[..., None]) * c + alpha[..., None] * color
        self.candidate_errors[y0:y1, x0:x1] = (t - new) ** 2
        gain = float(self.base_loss - self.candidate_errors.mean())
        self.candidate_errors[y0:y1, x0:x1] = self.errors[y0:y1, x0:x1]
        return gain, color, None


class VectorScene:
    def __init__(self, background, shapes):
        self.background, self.shapes = background, shapes

    def render(self, height, width):
        canvas = np.broadcast_to(self.background, (height, width, 3)).copy()
        for shape in self.shapes:
            a = (mask(shape, height, width, 3) * shape.alpha)[..., None]
            canvas = canvas * (1 - a) + a * shape.color
        return canvas

    def svg(self, width, height):
        rgb = lambda c: "rgb(" + ",".join(str(round(float(v) * 255)) for v in c) + ")"
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', f'<rect width="100%" height="100%" fill="{rgb(self.background)}"/>']
        for shape in self.shapes:
            p = shape.points * [width, height]
            lo, hi = p.min(0), p.max(0)
            style = f'fill="{rgb(shape.color)}" fill-opacity="{shape.alpha:.5f}"'
            if shape.kind == "triangle":
                coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in p)
                lines.append(f'<polygon points="{coords}" {style}/>')
            elif shape.kind == "rectangle":
                lines.append(f'<rect x="{lo[0]:.3f}" y="{lo[1]:.3f}" width="{hi[0]-lo[0]:.3f}" height="{hi[1]-lo[1]:.3f}" {style}/>')
            else:
                center, radius = (lo + hi) / 2, (hi - lo) / 2
                lines.append(f'<ellipse cx="{center[0]:.3f}" cy="{center[1]:.3f}" rx="{radius[0]:.3f}" ry="{radius[1]:.3f}" {style}/>')
        return "\n".join(lines + ['</svg>'])


def reconstruct(target, count=128, candidates=48, refinements=40, seed=7, model=None, callback=None, restarts=1, shape_family="mixed", backend='regional', min_triangle_angle=0, fixed_opacity=None):
    if min(count, candidates, refinements, restarts) < 1 or shape_family not in ('mixed', *KINDS):
        raise ValueError('Invalid reconstruction settings')
    if backend not in ('regional', 'cached', 'reference'):
        raise ValueError('Unknown scoring backend')
    if not 0 <= min_triangle_angle <= 30 or (fixed_opacity is not None and not 0 < fixed_opacity <= 1):
        raise ValueError('Invalid geometric style constraints')
    rng = np.random.default_rng(seed)
    background = target.mean((0, 1))
    canvas = np.broadcast_to(background, target.shape).copy()
    history, shapes = [float(np.mean((target - canvas) ** 2))], []
    evaluations = 0
    for step in range(count):
        context = ScoringContext(target, canvas, backend)
        best, best_gain = None, -np.inf
        for restart in range(restarts):
            pool = [proposal(target, canvas, rng, shape_family, context.probabilities if backend != 'reference' else None, min_triangle_angle, fixed_opacity) for _ in range(candidates * (4 if model is not None else 1))]
            if model is not None:
                predictions = model.predict([features(p, target, canvas, context.coarse if backend != 'reference' else None) for p in pool])
                pool = [pool[i] for i in np.argsort(predictions)[-candidates:]]
            local, local_gain = None, -np.inf
            for p in pool:
                gain, _, _ = context.evaluate(p)
                evaluations += 1
                if local is None or gain > local_gain:
                    local, local_gain = p, gain
            for iteration in range(refinements):
                # Change one geometric coordinate, allowing precise edge adjustments.
                p = mutate(local, rng, .08 * (.04 ** (iteration / max(1, refinements))), min_triangle_angle, fixed_opacity)
                gain, _, _ = context.evaluate(p)
                evaluations += 1
                if gain > local_gain:
                    local, local_gain = p, gain
            # Rank restart winners with the same antialiased renderer used at commit.
            # Full-resolution reference scoring guards every antialiased commit.
            gain, local_color, local_alpha = score(local, target, canvas, antialias=3, base_loss=context.base_loss)
            evaluations += 1
            if gain > best_gain:
                best, best_gain = local, gain
                color, alpha = local_color, local_alpha
        if best_gain > 0:
            best.color = color
            shapes.append(best)
            canvas = canvas * (1 - alpha[..., None]) + alpha[..., None] * color
        history.append(float(np.mean((target - canvas) ** 2)))
        if callback and (step % 4 == 0 or step == count - 1):
            callback(step + 1, history[-1], canvas.copy())
    return VectorScene(background, shapes), history, evaluations
