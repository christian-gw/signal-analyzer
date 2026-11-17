from importlib.metadata import version

import signal_analyzer


def test_package_version_matches_installed_distribution():
    assert signal_analyzer.__version__ == version("signal-analyzer")
