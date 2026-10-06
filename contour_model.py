"""Color-region vectorization and shared differentiable composition refinement."""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2lab
from skimage.measure import find_contours, approximate_polygon
from skimage.metrics import structural_similarity
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Region:
    contours: list
    color: np.ndarray


def region_mask(region, height, width, scale=3):
    image = Image.new('1', (width * scale, height * scale))
    for points in region.contours:
        layer = Image.new('1', image.size)
        coordinates = points * [width * scale, height * scale] - .5
        ImageDraw.Draw(layer).polygon([tuple(p) for p in coordinates], fill=1)
        image = ImageChops.logical_xor(image, layer)
    image = image.convert('L')
    if scale > 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255


class ContourScene:
    def __init__(self, background, regions):
        self.background, self.regions = np.asarray(background, dtype=np.float32), regions

    def render(self, height, width):
        canvas = np.broadcast_to(self.background, (height, width, 3)).copy()
        for region in self.regions:
            alpha = region_mask(region, height, width)[..., None]
            canvas = canvas * (1 - alpha) + alpha * region.color
        return canvas

    def svg(self, width, height):
        def rgb(color):
            return 'rgb(' + ','.join(str(round(float(v) * 255)) for v in color) + ')'
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 f'<rect width="100%" height="100%" fill="{rgb(self.background)}"/>']
        for region in self.regions:
            commands = []
            for contour in region.contours:
                points = contour * [width, height]
                commands.append(f'M {points[0, 0]:.4f},{points[0, 1]:.4f} ' + ' '.join(f'L {x:.4f},{y:.4f}' for x, y in points[1:]) + ' Z')
            path = ' '.join(commands)
            lines.append(f'<path d="{path}" fill="{rgb(region.color)}" fill-rule="evenodd"/>')
        return '\n'.join(lines + ['</svg>'])


def initialize_regions(target, count=128, palette=24, seed=7, tolerance=1.0):
    h, w = target.shape[:2]
    rgb = target.reshape(-1, 3)
    lab = rgb2lab(target).reshape(-1, 3)
    distinct = len(np.unique((rgb * 255).round().astype('uint8'), axis=0))
    clusters = min(palette, count + 1, distinct, len(rgb))
    with threadpool_limits(limits=4):
        model = MiniBatchKMeans(n_clusters=clusters, random_state=seed, n_init=3,
                               batch_size=2048, max_iter=100)
        labels = model.fit_predict(lab).reshape(h, w)
    # Mini-batch clustering can split a flat color across near-identical centers.
    # Merge those centers before tracing components, otherwise it creates seams.
    population = np.bincount(labels.ravel(), minlength=clusters)
    remap = np.zeros(clusters, dtype=np.int32)
    representatives = []
    for label in np.argsort(population)[::-1]:
        if population[label] == 0:
            continue
        center = model.cluster_centers_[label]
        distances = [np.linalg.norm(center - other) for other in representatives]
        if distances and min(distances) < 3:
            remap[label] = int(np.argmin(distances))
        else:
            remap[label] = len(representatives)
            representatives.append(center)
    labels = remap[labels]
    clusters = len(representatives)
    # Categorical majority, not an ordinal median of arbitrary cluster IDs.
    votes = np.stack([ndimage.convolve((labels == label).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8), mode='nearest') for label in range(clusters)])
    labels = votes.argmax(0)
    background_label = int(np.argmax(np.bincount(labels.ravel(), minlength=clusters)))
    background = target[labels == background_label].mean(0)
    groups = []
    component_count = 0
    # Four-neighbor connectivity avoids joining regions at a single diagonal pixel.
    for label in range(clusters):
        if label == background_label:
            continue
        connected, total = ndimage.label(labels == label)
        bounds = ndimage.find_objects(connected)
        contours = []
        for index, box in enumerate(bounds, 1):
            if box is None:
                continue
            inside = connected[box] == index
            if inside.sum() < 2:
                continue
            component_count += 1
            for contour in find_contours(np.pad(inside.astype(float), 1), .5):
                simplified = approximate_polygon(contour, tolerance=tolerance)
                if len(simplified) < 4:
                    simplified = approximate_polygon(contour, tolerance=.25)
                if len(simplified) < 4:
                    continue
                y = simplified[:, 0] - .5 + box[0].start
                x = simplified[:, 1] - .5 + box[1].start
                contours.append(np.stack([x / w, y / h], -1).astype(np.float32))
        if contours:
            pixels = target[labels == label]
            color = pixels.mean(0)
            importance = float(np.sum((pixels - background) ** 2))
            groups.append((importance, Region(contours, np.asarray(color, dtype=np.float32))))
    # All components of one palette color share a compound path, including holes.
    # This keeps small details instead of dropping them to meet a layer budget.
    groups.sort(key=lambda item: item[0], reverse=True)
    regions = [region for _, region in groups]
    return ContourScene(background, regions), dict(palette_clusters=clusters,
        available_regions=component_count, selected_regions=len(regions),
        traced_components=component_count,
        vertices=sum(len(c) for r in regions for c in r.contours))


class SharedRefinement(nn.Module):
    """Joint colors plus one smooth deformation shared by every region boundary."""
    def __init__(self, scene, height, width):
        super().__init__()
        self.scene = scene
        self.height, self.width = height, width
        masks = np.stack([region_mask(r, height, width) for r in scene.regions])
        transmission = np.ones((height, width), dtype=np.float32)
        weights = np.empty_like(masks)
        for index in range(len(masks) - 1, -1, -1):
            weights[index] = masks[index] * transmission
            transmission *= 1 - masks[index]
        self.register_buffer('weights', torch.from_numpy(weights))
        self.register_buffer('transmission', torch.from_numpy(transmission))
        colors = torch.tensor(np.stack([r.color for r in scene.regions])).clamp(.0001, .9999)
        self.colors = nn.Parameter(torch.logit(colors))
        self.background = nn.Parameter(torch.logit(torch.tensor(scene.background).clamp(.0001, .9999)))
        grid_size = max(2, min(8, min(height, width) // 6))
        self.displacement = nn.Parameter(torch.zeros(1, 2, grid_size, grid_size))
        max_pixels = min(2., min(height, width) / 10)
        self.register_buffer('limits', torch.tensor([2 * max_pixels / width, 2 * max_pixels / height]).reshape(1, 2, 1, 1))
        y, x = torch.meshgrid((torch.arange(height) + .5) * 2 / height - 1,
                              (torch.arange(width) + .5) * 2 / width - 1, indexing='ij')
        self.register_buffer('grid', torch.stack([x, y], -1)[None])

    def field(self):
        return self.displacement.tanh() * self.limits

    def forward(self):
        base = torch.einsum('nhw,nc->hwc', self.weights, self.colors.sigmoid())
        base = base + self.transmission[..., None] * self.background.sigmoid()
        field = F.interpolate(self.field(), size=(self.height, self.width), mode='bilinear', align_corners=False)
        grid = self.grid + field.permute(0, 2, 3, 1)
        return F.grid_sample(base.permute(2, 0, 1)[None], grid,
            align_corners=False, padding_mode='border')[0].permute(1, 2, 0)

    def export_scene(self):
        with torch.no_grad():
            colors = self.colors.sigmoid().numpy()
            field = self.field()
            regions = []
            for index, region in enumerate(self.scene.regions):
                contours = []
                for contour in region.contours:
                    if float(field.var(dim=(2, 3)).max()) > 1e-12:
                        dense = []
                        for start, end in zip(contour[:-1], contour[1:]):
                            length = np.linalg.norm((end - start) * [self.width, self.height])
                            samples = max(1, int(np.ceil(length / 8)))
                            dense.extend(start + (end - start) * t / samples for t in range(samples))
                        dense.append(contour[-1])
                        contour = np.asarray(dense, dtype=np.float32)
                    original = torch.from_numpy(contour) * 2 - 1
                    points = original.clone()
                    # Invert the sampling warp, preserving shared boundary locations.
                    for _ in range(6):
                        shifts = F.grid_sample(field, points[None, None], align_corners=False, padding_mode='border')[0, :, 0].T
                        points = original - shifts
                    contours.append(((points + 1) / 2).numpy())
                regions.append(Region(contours, colors[index].copy()))
            return ContourScene(self.background.sigmoid().numpy(), regions)


def structural_loss(predicted, target):
    """Differentiable windowed SSIM with the same sample covariance convention."""
    window = min(7, min(target.shape[:2]))
    window -= 1 - window % 2
    if window < 3:
        return (predicted - target).square().mean()
    a, b = predicted.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None]
    mean = lambda value: F.avg_pool2d(value, window, stride=1)
    ma, mb = mean(a), mean(b)
    correction = window ** 2 / (window ** 2 - 1)
    va = (mean(a*a) - ma*ma).clamp_min(0) * correction
    vb = (mean(b*b) - mb*mb).clamp_min(0) * correction
    covariance = (mean(a*b) - ma*mb) * correction
    luminance = (2*ma*mb + .01**2) / (ma*ma + mb*mb + .01**2)
    structure = (2*covariance + .03**2) / (va + vb + .03**2)
    return 1 - (luminance * structure).mean()


def contour_fit(target, count=128, palette=24, steps=80, seed=7, edge_weight=.2, callback=None):
    if min(count, palette) < 1 or steps < 0 or edge_weight < 0:
        raise ValueError('Invalid contour settings')
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 3 or target.shape[-1] != 3 or not np.isfinite(target).all():
        raise ValueError('Expected a finite RGB image')
    scene, metadata = initialize_regions(target, count, palette, seed)
    initial = scene.render(*target.shape[:2])
    best_mse = float(np.mean((initial - target) ** 2))
    window = min(7, min(target.shape[:2]))
    window = window if window % 2 else window - 1
    def ssim(image):
        return float(structural_similarity(target, image, channel_axis=-1,
            data_range=1, win_size=window)) if window >= 3 else None
    best_ssim = ssim(initial)
    history = [float(np.mean((target - target.mean((0, 1))) ** 2)), best_mse]
    metadata['initialized_mse'] = best_mse
    metadata['initialized_ssim'] = best_ssim
    metadata['initialized_vertices'] = metadata['vertices']
    if callback:
        callback(0, best_mse, initial)
    if scene.regions and steps:
        model = SharedRefinement(scene, *target.shape[:2])
        tensor = torch.from_numpy(target)
        optimizer = torch.optim.Adam([
            {'params': [model.colors, model.background], 'lr': .025},
            {'params': [model.displacement], 'lr': .006}])
        for step in range(steps):
            rendered = model()
            loss = (rendered - tensor).square().mean()
            loss = loss + .1 * structural_loss(rendered, tensor)
            for axis in (0, 1):
                if target.shape[axis] > 1:
                    loss = loss + edge_weight * (rendered.diff(dim=axis) - tensor.diff(dim=axis)).square().mean()
            field = model.field()
            loss = loss + .1 * (field.diff(dim=2).square().mean() + field.diff(dim=3).square().mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0 or step == steps - 1:
                candidate = model.export_scene()
                actual = candidate.render(*target.shape[:2])
                mse = float(np.mean((actual - target) ** 2))
                candidate_ssim = ssim(actual)
                # The exported vector geometry, not a soft proxy, decides acceptance.
                if mse < best_mse and (best_ssim is None or candidate_ssim >= best_ssim):
                    scene, best_mse, best_ssim = candidate, mse, candidate_ssim
            history.append(best_mse)
            if callback and ((step + 1) % 10 == 0 or step == steps - 1):
                callback(step + 1, best_mse, scene.render(*target.shape[:2]))
    metadata.update(best_mse=best_mse, best_ssim=best_ssim, refinement_steps=steps if scene.regions else 0,
                    vertices=sum(len(c) for r in scene.regions for c in r.contours),
                    method='Lab MiniBatchKMeans + connected contours + shared PyTorch refinement')
    return scene, history, metadata


def main():
    import argparse
    import json
    from pathlib import Path
    from PIL import ImageOps
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/contours'))
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--layers', type=int, default=128)
    parser.add_argument('--palette', type=int, default=24)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if min(args.size, args.layers, args.palette) < 1 or args.steps < 0:
        parser.error('Size, layers, and palette must be positive; steps must be nonnegative')
    torch.set_num_threads(4)
    image = ImageOps.exif_transpose(Image.open(args.input)).convert('RGBA')
    image = Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image).convert('RGB')
    original_size = image.size
    image.thumbnail((args.size, args.size))
    target = np.asarray(image, dtype=np.float32) / 255
    def progress(step, mse, rendered):
        print(f'{step}/{args.steps} exported MSE={mse:.6f}', flush=True)
    scene, history, metadata = contour_fit(target, args.layers, args.palette, args.steps, args.seed, callback=progress)
    args.output.mkdir(exist_ok=True, parents=True)
    width, height = original_size
    factor = 1024 / max(width, height)
    pixels = scene.render(max(1, round(height*factor)), max(1, round(width*factor)))
    Image.fromarray((pixels.clip(0, 1)*255).astype('uint8')).save(args.output / 'result.png')
    (args.output / 'result.svg').write_text(scene.svg(width, height))
    metadata.update(loss_history=history, settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    (args.output / 'metrics.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
