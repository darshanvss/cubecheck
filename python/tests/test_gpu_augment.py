import cv2
import numpy as np
import torch

from cubecheck import gpu_augment as G
from cubecheck.corners import view_matrix, warp


def picture(h: int = 720, w: int = 1280) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(60):  # coloured blocks with sharp edges
        x, y = rng.integers(0, w - 80), rng.integers(0, h - 80)
        img[y:y + rng.integers(20, 80), x:x + rng.integers(20, 80)] = rng.integers(0, 256, 3)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def as_batch(arr) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr))[None]


def test_gpu_view_matches_cpu_warp():
    img = picture()
    for m in (view_matrix(np.array([600.0, 300.0]), 300, angle=20, flip=True), view_matrix(np.array([640.0, 360.0]), 1400, angle=-10)):
        canvas, info = G.cut_canvas(img, m)
        x = G.cut_view(as_batch(canvas).permute(0, 3, 1, 2).float() / 255, as_batch(info["theta"]), as_batch(info["filled"]))
        ours = (x[0].permute(1, 2, 0).numpy() * 255).round().astype(int)[..., ::-1]
        assert np.abs(ours - warp(img, m).astype(int)).mean() < 3


def test_gpu_cut_out_matches_cpu_cut_out():
    img = picture()
    m = view_matrix(np.array([500.0, 400.0]), 900, angle=15)
    canvas, info = G.cut_canvas(img, m)
    frame = torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).permute(2, 0, 1)
    gpu = G.cut_canvas_gpu([frame], as_batch(info["box"]))
    assert np.abs((gpu[0].permute(1, 2, 0).numpy() * 255).round() - canvas).mean() < 1.5


def test_jpeg_simulation_is_close_to_real_jpeg():
    crop = picture()[:G.CANVAS, :G.CANVAS]
    real = cv2.imdecode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], cv2.IMREAD_COLOR)
    x = torch.from_numpy(np.ascontiguousarray(crop[..., ::-1])).permute(2, 0, 1)[None].float() / 255
    ours = (G.jpeg(x, torch.tensor([40.0]))[0].permute(1, 2, 0).numpy() * 255).round()[..., ::-1]
    change = np.abs(real.astype(float) - crop).mean()
    assert np.abs(ours - real).mean() < 0.5 * change


def test_effect_settings_stack_and_off_means_unchanged():
    batch = {f"fx_{k}": as_batch(v) for k, v in G.draw_effects(np.random.default_rng(1), clean_chance=1.0).items()}
    x = torch.rand(1, 3, G.CANVAS, G.CANVAS)
    frame = torch.tensor([[0, 0, 1, 1, G.CANVAS, G.CANVAS]], dtype=torch.float32)
    out = G.apply_effects(x, {k[3:]: v for k, v in batch.items()}, frame)
    assert torch.allclose(out, (x * 255).round() / 255, atol=1e-5)
    shapes = {k: v.shape for k, v in G.draw_effects(np.random.default_rng(2), k=2.5).items()}
    assert shapes == {k[3:]: v.shape[1:] for k, v in batch.items()}
