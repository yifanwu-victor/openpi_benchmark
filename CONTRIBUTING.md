# Development

Use the setup and regression commands in [README.md](README.md). Keep changes focused on the AnyTouch2 concat baseline.

- Keep `pyproject.toml` and `uv.lock` in sync; run `uv lock --check`.
- Run `ruff check .` and format changed Python files.
- Run the CPU regression suite, and test HDF5 conversion when changing data semantics.
- Check existing checkpoint compatibility when changing model parameters or token layout.
- Never commit datasets, pretrained weights, training checkpoints, local credentials, or experiment logs.

The GitHub workflow performs static checks. Runtime tests are run in the development environment described in the README.
