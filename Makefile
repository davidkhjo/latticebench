# Developer shortcuts. The version lives only in src/latticebench/__init__.py
# (hatchling reads it); these targets bump it, commit, and tag in one step.

.PHONY: test lint typecheck cov check reproduce bump-patch bump-minor bump-major

test:       ; uv run pytest
lint:       ; uv run ruff check . && uv run ruff format --check .
typecheck:  ; uv run mypy
cov:        ; uv run pytest --cov --cov-report=term-missing
check: lint typecheck test

# Retrain the model and regenerate every figure and table from the seed manifests.
# Needs the optional extras: uv pip install -e ".[ebm,viz,llm-local]"
reproduce:
	uv run python experiments/train_solver.py
	uv run python experiments/run.py
	uv run python experiments/figures.py

# make bump-patch   # 0.1.0 -> 0.1.1  (also: bump-minor, bump-major)
# Update CHANGELOG.md first; after bumping, `git push --follow-tags` and cut a
# release with `gh release create vX.Y.Z --title vX.Y.Z`.
bump-patch: ; @$(MAKE) _bump PART=patch
bump-minor: ; @$(MAKE) _bump PART=minor
bump-major: ; @$(MAKE) _bump PART=major

_bump:
	uv run hatch version $(PART)
	@V=$$(uv run hatch version); \
	git commit -am "v$$V"; \
	git tag "v$$V"; \
	echo "bumped to v$$V — run: git push --follow-tags"
