# Packaging and local installation

The project uses a `src` layout so imports resolve through the installed
package rather than accidentally finding modules in the repository root. The
distribution name is `signal-analyzer`; the Python package is
`signal_analyzer`.

## Development installation

Create or activate a virtual environment, then run from the repository root:

```text
python -m pip install -e ".[test,examples]"
```

An editable installation reflects Python source edits without reinstalling.
Reinstall after changing package metadata or dependencies in `pyproject.toml`.

## Application or GUI installation

Use a regular installation when checking how an external application receives
the project:

```text
python -m pip install .
```

Then verify imports from a directory outside this repository:

```python
from signal_analyzer import Analysis, AnalysisResult, Measurement, Signal
```

A regular installation is intentionally part of release validation because an
editable install can expose source-tree behavior that a built distribution
does not.

## Build without publishing

Install the standard build frontend if needed, then build both distribution
formats:

```text
python -m pip install build
python -m build
```

This creates a wheel and source archive under `dist/`. Building does not upload
anything. Before treating a version as complete, install the wheel into a clean
temporary environment, import `signal_analyzer` outside the repository, and run
a small measurement workflow.

## Dependency groups

- Runtime: NumPy, SciPy, and pandas.
- `test`: pytest.
- `examples`: Jupyter execution support and matplotlib.

Readers for private formats, GUI dependencies, persistence backends, and
application plotting do not belong in the core dependency set.

## References

- [Python Packaging User Guide: Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Python Packaging User Guide: src layout versus flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [pip: local project installs](https://pip.pypa.io/en/stable/topics/local-project-installs/)
