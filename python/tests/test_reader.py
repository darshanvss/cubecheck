import numpy as np
import torch

from cubecheck.reader import FACE_PX, StickerReader, face_maps, to_tensor
from cubecheck.verify import cell_centres


def test_face_maps_put_each_sticker_in_its_grid_cell():
    # Paint a dot at each of the 27 sticker centres with its own colour: after
    # straightening, cell n of the map must hold colour n.
    kp = np.float32([[300, 60], [480, 160], [470, 360], [300, 460], [130, 360], [120, 160], [300, 260]])
    img = np.zeros((520, 600, 3), np.uint8)
    colours = [(n * 9 % 256, n * 37 % 256, 255 - n * 9) for n in range(27)]
    for (x, y), c in zip(cell_centres(kp), colours):
        img[int(y) - 4:int(y) + 5, int(x) - 4:int(x) + 5] = c
    maps = face_maps(img, kp)
    step = FACE_PX // 3
    for n, c in enumerate(colours):
        k, j, i = n // 9, n % 9 // 3, n % 3
        rgb = maps[k, j * step + step // 2, i * step + step // 2]
        assert tuple(int(v) for v in rgb[::-1]) == c  # maps are RGB, the image BGR


def test_reader_gives_7_answers_for_each_of_27_stickers():
    model = StickerReader().eval()
    maps = np.zeros((2, 3, FACE_PX, FACE_PX, 3), np.uint8)
    assert model(to_tensor(maps)).shape == (2, 27, 7)
    assert torch.allclose(torch.softmax(model(to_tensor(maps)), -1).sum(-1), torch.ones(2, 27))
