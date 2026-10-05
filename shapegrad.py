"""Fit editable geometric art using differentiable rendering."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn


class ShapeScene(nn.Module):
    def __init__(self, target, count=64, seed=7):
        super().__init__()
        generator = torch.Generator(device=target.device).manual_seed(seed)
        centers = torch.rand(count, 2, generator=generator, device=target.device)
        h, w, _ = target.shape
        colors = target[(centers[:, 1] * (h - 1)).long(),
                        (centers[:, 0] * (w - 1)).long()].clamp(.01, .99)
        self.centers = nn.Parameter(torch.logit(centers.clamp(.01, .99)))
        self.radii = nn.Parameter(torch.full_like(centers, -2.0))
        self.colors = nn.Parameter(torch.logit(colors))
        self.opacity = nn.Parameter(torch.zeros(count, device=target.device))
        self.register_buffer('background', target.mean((0, 1)))

    def forward(self, height, width, sharpness=60):
        y, x = torch.meshgrid(
            (torch.arange(height, device=self.centers.device) + .5) / height,
            (torch.arange(width, device=self.centers.device) + .5) / width,
            indexing='ij')
        grid = torch.stack((x, y), -1)
        canvas = self.background.expand(height, width, 3)
        for center, radius, color, opacity in zip(
                self.centers.sigmoid(), .005 + .495 * self.radii.sigmoid(),
                self.colors.sigmoid(), self.opacity.sigmoid()):
            distance = (((grid - center) / radius) ** 2).sum(-1)
            alpha = (torch.sigmoid((1 - distance) * sharpness) * opacity)[..., None]
            canvas = canvas * (1 - alpha) + color * alpha
        return canvas

    def svg(self, width, height):
        def rgb(value):
            return 'rgb(' + ','.join(str(round(float(v) * 255)) for v in value) + ')'
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 f'<rect width="100%" height="100%" fill="{rgb(self.background)}"/>']
        with torch.no_grad():
            for c, r, color, a in zip(self.centers.sigmoid(),
                    .005 + .495 * self.radii.sigmoid(), self.colors.sigmoid(), self.opacity.sigmoid()):
                lines.append(f'<ellipse cx="{float(c[0])*width:.3f}" cy="{float(c[1])*height:.3f}" rx="{float(r[0])*width:.3f}" ry="{float(r[1])*height:.3f}" fill="{rgb(color)}" fill-opacity="{float(a):.5f}"/>')
        return '\n'.join(lines + ['</svg>'])


def objective(rendered, target, edge_weight=.2):
    loss = (rendered - target).square().mean()
    for axis in (0, 1):
        if target.shape[axis] > 1:
            loss = loss + edge_weight * (rendered.diff(dim=axis) - target.diff(dim=axis)).square().mean()
    return loss


def accept_proposal(delta, temperature, random_value):
    return delta <= 0 or (temperature > 0 and random_value < math.exp(-delta / temperature))


def fit(target, count=64, steps=500, seed=7, edge_weight=.2, callback=None,
        method='adam', temperature=.001):
    if method not in ('adam', 'hill', 'anneal'):
        raise ValueError('Unknown optimization method')
    if min(count, steps) < 1 or edge_weight < 0 or temperature <= 0:
        raise ValueError('Invalid fit settings')
    scene = ShapeScene(target, count, seed)
    generator = torch.Generator(device=target.device).manual_seed(seed + 1)
    optimizer = torch.optim.Adam(scene.parameters(), lr=.035)
    history = []
    best_loss, best_state = float('inf'), None
    parameters = list(scene.parameters())
    def evaluate():
        rendered = scene(*target.shape[:2])
        return rendered, objective(rendered, target, edge_weight)
    for step in range(steps + 1):
        with torch.set_grad_enabled(method == 'adam'):
            rendered, loss = evaluate()
        value = float(loss.detach())
        history.append(value)
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in scene.state_dict().items()}
        if callback and (step % 25 == 0 or step == steps):
            callback(step, value, rendered.detach())
        if step == steps:
            break
        if method == 'adam':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                index = int(torch.randint(count, (1,), generator=generator, device=target.device))
                old = [p[index].clone() for p in parameters]
                scale = .35 * (1 - .8 * step / steps)
                for p in parameters:
                    p[index].add_(torch.randn(p[index].shape, generator=generator,
                                            device=target.device) * scale)
                _, proposal = evaluate()
                delta = float(proposal) - value
                cooling = temperature * (.01 ** (step / steps)) if method == 'anneal' else 0
                draw = float(torch.rand((), generator=generator, device=target.device))
                if not accept_proposal(delta, cooling, draw):
                    for p, saved in zip(parameters, old):
                        p[index].copy_(saved)
    scene.load_state_dict(best_state)
    return scene, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--shapes', type=int, default=64)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--edge-weight', type=float, default=.2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--gif', action='store_true')
    parser.add_argument('--method', choices=['adam', 'hill', 'anneal'], default='adam')
    args = parser.parse_args()
    if min(args.shapes, args.steps, args.size) < 1 or args.edge_weight < 0:
        parser.error('shapes, steps, size must be positive; edge-weight must be nonnegative')
    torch.set_num_threads(4)
    image = ImageOps.exif_transpose(Image.open(args.input)).convert('RGBA')
    background = Image.new('RGBA', image.size, 'white')
    image = Image.alpha_composite(background, image).convert('RGB')
    original_size = image.size
    image.thumbnail((args.size, args.size))
    target = torch.tensor(np.asarray(image).copy(), dtype=torch.float32, device=args.device) / 255
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    def progress(step, loss, rendered):
        print(f'{step:4d}/{args.steps} loss={loss:.6f}')
        if args.gif:
            frames.append(Image.fromarray((rendered.clamp(0, 1).cpu().numpy()*255).astype('uint8')))
    scene, history = fit(target, args.shapes, args.steps, args.seed, args.edge_weight, progress, method=args.method)
    with torch.no_grad():
        rendered = scene(*target.shape[:2]).cpu().numpy()
    result = Image.fromarray((rendered.clip(0, 1)*255).astype('uint8'))
    result.save(args.output / 'result.png')
    (args.output / 'result.svg').write_text(scene.svg(*original_size), encoding='utf-8')
    (args.output / 'metrics.json').write_text(json.dumps({'initial_loss': history[0], 'best_loss': min(history), 'loss_history': history, 'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2), encoding='utf-8')
    if frames:
        frames.append(result)
        frames[0].save(args.output / 'progress.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f'Saved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
