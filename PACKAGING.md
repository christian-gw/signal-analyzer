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

## Development from a separate application

Keep application code in a separate repository so file readers, persistence,
plots, and user-interface dependencies do not leak into the numerical core. A
convenient local layout is:

```text
development/
|-- signal_analyzer/
`-- analysis_application/
```

Create the virtual environment in the application repository, activate it,
then install this package from the sibling checkout:

```text
python -m pip install -e "../signal_analyzer"
```

Use `-e "../signal_analyzer[test,examples]"` only when that environment also
needs to change and validate the core. Editable installs are a development
convenience, not a deployment format.

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

For CI and deployed applications, prefer a versioned wheel over installing a
mutable checkout:

```text
python -m pip install path/to/signal_analyzer-0.1.0-py3-none-any.whl
```

Until the project has a package-index release process, the application can
build that wheel from a pinned tag or commit and retain it as a CI artifact.
The application should record or lock the exact core version and its complete
environment.

## Build without publishing

Install the standard build frontend if needed, then build both distribution
formats:

```text
python -m pip install build twine
python -m build
python -m twine check dist/*
```

This creates a wheel and source archive under `dist/`. Building does not upload
anything. Before treating a version as complete, install the wheel into a clean
temporary environment, import `signal_analyzer` outside the repository, and run
a small measurement workflow.

The package version is declared in `pyproject.toml`. The public
`signal_analyzer.__version__` value reads the installed distribution metadata,
so a release has only one version value to update.

## Dependency groups

- Runtime: NumPy 1.23.5+, SciPy 1.10.1+, and pandas 1.5.3+.
- `test`: pytest.
- `examples`: Jupyter execution support and matplotlib.

The lower bounds are validated together on Python 3.10 in CI. The main test
matrix uses the normal latest-compatible dependency resolution on every
supported Python version. Upper bounds should be added only for a demonstrated
incompatibility. A consuming application should use its own lock file when it
needs a fully reproducible environment.

Readers for private formats, GUI dependencies, persistence backends, and
application plotting do not belong in the core dependency set.

## Release boundary

The package is pre-1.0, so consuming applications should pin a version rather
than assume compatibility across releases. Choose a publication target only
when external installation requires one: repository release artifacts, a
private package index, or PyPI. Tag-triggered publication automation should be
added after that destination and its credentials are defined.

## References

- [Python Packaging User Guide: Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Python Packaging User Guide: src layout versus flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [pip: local project installs](https://pip.pypa.io/en/stable/topics/local-project-installs/)
