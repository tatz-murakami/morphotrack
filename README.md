# morphotrack

Morphotrack version 2.

## Installation

```bash
pip install -e ".[dev]"
```

## Layout

- `src/morphotrack/` — the Python package (importable modules)
- `notebooks/` — analysis notebooks, numbered in pipeline order
- `tests/` — unit tests (`pytest`)
- `data/` — input data (not tracked by git)

## Usage in notebooks

After the editable install, notebooks can simply `import morphotrack`.
Add this to the top of each notebook so package edits are picked up without
restarting the kernel:

```python
%load_ext autoreload
%autoreload 2
```
