import numpy as np
import torch

from cubecheck.corners import IN_PX, CornerNet, apply, cube_shaped, full_view, invert, refine_view, soft_argmax, view_matrix, warp


def test_views_map_back_to_the_image():
    pts = np.array([[100.0, 50.0], [640.0, 360.0], [1200.0, 700.0]])
    for m in (full_view((720, 1280)), view_matrix(np.array([600.0, 300.0]), 400, angle=25, flip=True), refine_view(pts)):
        assert np.allclose(apply(invert(m), apply(m, pts)), pts, atol=1e-6)


def test_refine_view_puts_the_cube_in_the_middle_with_room_around():
    kp = np.array([[500.0, 200.0], [700.0, 260.0], [690.0, 480.0], [520.0, 520.0], [400.0, 420.0], [420.0, 250.0], [560.0, 360.0]])
    inside = apply(refine_view(kp), kp)
    assert inside.min() > 0.1 * IN_PX and inside.max() < 0.9 * IN_PX


def test_cube_shape_check():
    kp = np.array([[500.0, 200.0], [700.0, 260.0], [690.0, 480.0], [520.0, 520.0], [400.0, 420.0], [420.0, 250.0], [560.0, 360.0]])
    assert cube_shaped(kp)
    assert cube_shaped(kp * [-1, 1])  # mirrored: still a cube, turning the other way
    assert not cube_shaped(kp[[0, 1, 2, 3, 4, 5, 6]][[2, 1, 0, 3, 4, 5, 6]])  # two corners swapped
    assert not cube_shaped(np.where(np.arange(7)[:, None] == 6, [800.0, 360.0], kp))  # centre outside


def test_warp_shows_the_right_spot():
    img = np.zeros((720, 1280, 3), np.uint8)
    img[290:310, 890:910] = 255  # a white square centred at (900, 300)
    m = view_matrix(np.array([900.0, 300.0]), 256)
    crop = warp(img, m)
    ys, xs = np.nonzero(crop[..., 0] > 128)
    assert abs(xs.mean() - IN_PX / 2) < 1.5 and abs(ys.mean() - IN_PX / 2) < 1.5


def test_soft_argmax_finds_a_peak_between_grid_cells():
    heat = torch.full((1, 7, 64, 64), -50.0)
    heat[0, :, 10, 20] = heat[0, :, 10, 21] = 50.0  # two equal cells: the answer is halfway
    xy = soft_argmax(heat)[0, 0]
    assert torch.allclose(xy, torch.tensor([21.0 / 64, 10.5 / 64]), atol=1e-4)


def test_corner_net_shapes():
    heat, present = CornerNet().eval()(torch.zeros(2, 3, IN_PX, IN_PX))
    assert heat.shape == (2, 7, 64, 64) and present.shape == (2,)
