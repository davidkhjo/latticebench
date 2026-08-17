# Contributing

Thanks for your interest in LatticeBench.

## Development

```bash
uv venv
uv pip install -e ".[dev]"
make check          # ruff + mypy + pytest
```

The test suite is CPU-only and fast; most tests fix the grid at `n, m <= 4` so
uniqueness and energy properties can be checked exhaustively.

## Pull requests

- Branch from `main` and keep each change focused.
- Add or update tests, and make sure `make check` is green.
- CI runs on Python 3.10 through 3.13.

## Submitting benchmark results

To add a model to the leaderboard, run it against a published seed manifest and
open a pull request with the resulting JSON. The manifest is re-derived and the
puzzles re-graded on our side, so results cannot be gamed by editing instances.
See `SUBMISSION.md`.

## Releases

Update `CHANGELOG.md`, then `make bump-patch` (or `bump-minor` / `bump-major`),
`git push --follow-tags`, and `gh release create vX.Y.Z --title vX.Y.Z`, which
publishes to PyPI.
