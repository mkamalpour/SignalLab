from pathlib import Path
import pytest
from signallab.storage import atomic_target


def test_export_failure_keeps_previous_file_and_cleans_staging(tmp_path):
    target = tmp_path / "work.csv"
    target.write_bytes(b"previous completed export")
    with pytest.raises(OSError):
        with atomic_target(target) as temporary:
            temporary.write_bytes(b"incomplete")
            raise OSError("simulated disk failure")
    assert target.read_bytes() == b"previous completed export"
    assert list(tmp_path.iterdir()) == [target]


def test_successful_export_replaces_only_target(tmp_path):
    target = tmp_path / "work.csv"
    target.write_bytes(b"previous")
    with atomic_target(target) as temporary:
        temporary.write_bytes(b"complete")
    assert target.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [target]


def test_locked_destination_preserves_previous_export(tmp_path, monkeypatch):
    import signallab.storage as storage
    target = tmp_path / "work.csv"
    target.write_bytes(b"previous")
    def fail(*args):
        raise PermissionError("locked file")
    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(PermissionError):
        with atomic_target(target) as temporary:
            temporary.write_bytes(b"complete")
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]
