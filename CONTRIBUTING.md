# Contributing

Thanks for helping. Bug reports, wrong verdicts, ideas and pull requests are all welcome.

## Reporting a wrong verdict

Open an issue with the "Wrong verdict" template: the scramble, what CubeCheck said, the browser and device, and the light. A screenshot helps, but **never post a scramble that is still going to be used in a competition**.

## Making a change

1. Open an issue first for anything bigger than a small fix, so the approach can be agreed before you spend time on it.
2. Keep pull requests focused on one change, and describe what changed and how you checked it.
3. Run the checks before you push:

```sh
cd python && uv run python -m pytest
```

```sh
cd web && npm run typecheck && npm test
```

The check runs in two languages: Python (`python/cubecheck/verify.py`) and TypeScript (`web/src/lib/check.ts`). A change to one needs the same change in the other; `web/tests/check.test.ts` compares their verdicts.

## Licence

By contributing, you agree that your contribution is licensed under the [AGPL-3.0-or-later](LICENSE), like the rest of the project.

Please follow the [code of conduct](CODE_OF_CONDUCT.md).
