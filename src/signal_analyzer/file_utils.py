"""Small file-discovery helper."""

from pathlib import Path


def find_files(folder, pattern="*"):
    """Return matching files recursively, sorted by filename (case-insensitive).

    Returns a list of pathlib.Path objects. Filename is the primary sort key;
    identical case-insensitive filenames are ordered by full path so results do
    not depend on filesystem discovery order.

    Example
    -------
    >>> files = find_files(r"C:\\Data", "*.csv")
    """
    return sorted(
        (p for p in Path(folder).rglob(pattern) if p.is_file()),
        key=lambda p: (
            p.name.casefold(),
            p.as_posix().casefold(),
            p.as_posix(),
        ),
    )
