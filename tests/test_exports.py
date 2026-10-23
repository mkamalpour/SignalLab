"""Exercise actual desktop export handlers without a display or private data."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from signallab import analyzer


@pytest.mark.parametrize("kind", ["features", "model", "log"])
def test_desktop_failed_save_keeps_previous_and_reports_error(tmp_path, monkeypatch, kind):
    target = tmp_path / "saved.txt"
    target.write_bytes(b"previous completed export")
    notices, errors = [], []
    monkeypatch.setattr(analyzer.filedialog, "asksaveasfilename", lambda **kw: str(target))
    monkeypatch.setattr(analyzer.messagebox, "showinfo", lambda *a: notices.append(a))
    monkeypatch.setattr(analyzer.messagebox, "showerror", lambda *a: errors.append(a))
    app = SimpleNamespace(features=np.array([[1., 2.]]), feature_names=["a", "b"], labels=np.array([0]),
        model=object(), pipeline=object(), _model_names=[], _model_config=None, _model_length=4,
        _check_busy=lambda: False, _log=lambda *a: None, log_widget=SimpleNamespace(get=lambda *a: "log"))
    def fail(*args, **kwargs):
        if kind == "features": args[1].write_bytes(b"partial")
        elif kind == "model": args[1].write(b"partial")
        else: args[0].write_bytes(b"partial")
        raise OSError("simulated disk full")
    if kind == "features":
        monkeypatch.setattr(pd.DataFrame, "to_csv", fail)
        analyzer.SignalAnalyzerApp._export_features(app)
    elif kind == "model":
        monkeypatch.setattr(analyzer.core, "model_package", lambda *a: {})
        monkeypatch.setattr(analyzer.pickle, "dump", fail)
        analyzer.SignalAnalyzerApp._export_model(app)
    else:
        from pathlib import Path
        monkeypatch.setattr(Path, "write_text", fail)
        analyzer.SignalAnalyzerApp._save_log(app)
    assert errors and not notices
    assert target.read_bytes() == b"previous completed export"
    assert list(tmp_path.iterdir()) == [target]
