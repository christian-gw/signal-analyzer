from pathlib import Path

from signal_analyzer.file_utils import find_files


def test_find_files_recurses_filters_and_sorts_by_filename(tmp_path):
    first_dir = tmp_path / "a_directory"
    second_dir = tmp_path / "z_directory"
    first_dir.mkdir()
    second_dir.mkdir()

    bravo = first_dir / "Bravo.csv"
    alpha = second_dir / "alpha.csv"
    charlie = tmp_path / "charlie.csv"
    ignored = first_dir / "ignored.txt"
    for path in (bravo, alpha, charlie, ignored):
        path.touch()

    files = find_files(tmp_path, "*.csv")

    # Filename order differs from full-path order, and the mixed case checks
    # that sorting is case-insensitive.
    assert files == [alpha, bravo, charlie]
    assert all(isinstance(path, Path) for path in files)


def test_find_files_excludes_matching_directories(tmp_path):
    matching_directory = tmp_path / "directory.csv"
    matching_directory.mkdir()
    expected = matching_directory / "sample.csv"
    expected.touch()

    assert find_files(tmp_path, "*.csv") == [expected]
