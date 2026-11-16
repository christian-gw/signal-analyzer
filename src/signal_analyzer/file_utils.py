"""Small file-discovery helper."""

from pathlib import Path


def find_files(folder, pattern="*"):
    """Return matching files recursively, sorted by filename (case-insensitive).

    Returns a list of pathlib.Path objects. Sort key is p.name.lower(),
    not the full path. Files with identical names retain discovery order.

    Example
    -------
    >>> files = find_files(r"C:\\Data", "*.csv")
    """
    return sorted(
        (p for p in Path(folder).rglob(pattern) if p.is_file()),
        key=lambda p: p.name.lower(),
    )
