# cubecheck (Python)

The check library and the training code: the corner model (`corners.py`), the sticker reader (`reader.py`), the cube rules and slip test (`verify.py`), and the scripts that train and evaluate them. Skewb, pyraminx and FTO (experimental) are in `puzzles.py` (their shapes in `puzzle_data/`), `shapes.py` and the `poly_*.py` files ([docs/other-puzzles.md](../docs/other-puzzles.md)). See the [main README](../README.md).

```sh
uv sync --extra train
uv run python -m pytest
```
