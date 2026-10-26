import pickle

import numpy as np
import pytest
from scipy.io import savemat
from sklearn.feature_selection import SelectKBest

from signallab import core


@pytest.mark.parametrize("length", [127, 128, 1024])
def test_parseval_power_and_band(length):
    rng = np.random.default_rng(8)
    s = rng.normal(size=length)
    f, p = core.spectrum(s, 1000)
    assert np.sum(p) == pytest.approx(np.mean(s * s))
    features = core.frequency_features(s, core.FeatureConfig(1000, 0, 500))
    assert features["band_power"] == pytest.approx(np.mean(s * s))
    assert len(f) == length // 2 + 1


def test_known_sinusoid():
    s = np.sin(2 * np.pi * 32 * np.arange(1024) / 1024)
    f = core.frequency_features(s, core.FeatureConfig(1024, 30, 34))
    assert f["dom_freq"] == 32
    assert f["fft_power"] == pytest.approx(0.5)
    assert f["band_power"] == pytest.approx(0.5)
    assert f["spec_centroid"] == pytest.approx(32)
    assert (
        core.frequency_features(s, core.FeatureConfig(1024, 100, 200))["band_power"]
        < 1e-20
    )


@pytest.mark.parametrize("value", [0.0, 3.0])
def test_constant_signals_are_finite(value):
    features = core.extract_features(
        np.full((3, 128), value), core.FEATURE_NAMES, core.FeatureConfig()
    )
    assert np.isfinite(features).all()
    assert features[0, 3] == 0 and features[0, 4] == 0


@pytest.mark.parametrize(
    "array",
    [
        np.array([1, np.nan, 3, 4]),
        np.array([1, np.inf, 3, 4]),
        np.ones((2, 3, 4)),
        np.array(["a", "b", "c", "d"]),
        np.ones(4) * 1j,
        np.ones(3),
    ],
)
def test_invalid_signals_rejected(array):
    with pytest.raises(ValueError):
        core.signal_matrix(array)


@pytest.mark.parametrize(
    "args", [(0, 0, 1), (100, 0, 51), (100, 4, 4), (100, -1, 10), (np.nan, 0, 1)]
)
def test_invalid_frequency_configuration(args):
    with pytest.raises(ValueError):
        core.FeatureConfig(*args)


def test_mat_class_matrices_preserve_labels_and_samples(tmp_path):
    path = tmp_path / "class.mat"
    x1 = np.arange(6 * 1024).reshape(6, 1024)
    x2 = np.ones((1024, 7))
    savemat(path, {"X1": x1, "X2": x2})
    X, y = core.labelled_mat(path)
    assert X.shape == (13, 1024)
    np.testing.assert_array_equal(X[:6], x1)
    assert set(y) == {1, 2}


def test_explicit_mat_transpose_and_noncontiguous_labels(tmp_path):
    path = tmp_path / "labelled.mat"
    savemat(path, {"signals": np.ones((128, 12)), "labels": np.repeat([10, 30], 6)})
    X, y = core.labelled_mat(path, "signals", "labels")
    assert X.shape == (12, 128)
    assert set(y) == {10, 30}


def test_mat_never_invents_labels(tmp_path):
    path = tmp_path / "unlabelled.mat"
    savemat(path, {"X": np.ones((12, 128))})
    with pytest.raises(ValueError, match="labels are missing"):
        core.labelled_mat(path)


def test_mat_rejects_different_lengths(tmp_path):
    path = tmp_path / "length.mat"
    savemat(path, {"X1": np.ones((6, 128)), "X2": np.ones((6, 64))})
    with pytest.raises(ValueError, match="different signal lengths"):
        core.labelled_mat(path)


@pytest.fixture
def training_data():
    rng = np.random.default_rng(4)
    t = np.arange(128) / 128
    rows = [
        np.sin(2 * np.pi * frequency * t) + rng.normal(0, 0.05, 128)
        for frequency in [8] * 30 + [25] * 30
    ]
    names = ["mean", "variance", "dom_freq", "spec_centroid"]
    return (
        core.extract_features(rows, names, core.FeatureConfig(128, 0, 64)),
        np.repeat([10, 30], 30),
        rows,
        names,
    )


@pytest.mark.parametrize("model", ["svm", "random_forest", "logistic"])
@pytest.mark.parametrize("scaler", ["standard", "minmax", "none"])
def test_each_classifier_and_scaler(training_data, model, scaler):
    X, y, rows, names = training_data
    result = core.train(X, y, dict(model=model, scaler=scaler, k=2))
    assert result["accuracy"] >= 0.9
    assert set(result["pipeline"].classes_) == {10, 30}
    assert result["train_count"] == 48 and result["test_count"] == 12
    assert result['balanced_accuracy'] >= .9
    assert result['macro_f1'] >= .9
    assert result['majority_baseline_accuracy'] == .5
    assert result['majority_baseline_class'] in {10, 30}


def test_grid_search_fits_preprocessing_inside_folds(training_data, monkeypatch):
    X, y, _, _ = training_data
    fits = []
    original = SelectKBest.fit

    def observed(self, x, y):
        fits.append(len(x))
        return original(self, x, y)

    monkeypatch.setattr(SelectKBest, "fit", observed)
    result = core.train(X, y, dict(k=2, grid=True, folds=3))
    assert result["best_params"]
    assert min(fits) == 32
    assert max(fits) == 48


def test_incremental_selection_reserves_holdout(training_data, monkeypatch):
    X, y, _, _ = training_data
    fits = []
    original = SelectKBest.fit

    def observed(self, x, y):
        fits.append(len(x))
        return original(self, x, y)

    monkeypatch.setattr(SelectKBest, "fit", observed)
    result = core.incremental_evaluation(X, y, folds=3)
    assert len(result["scores"]) == X.shape[1]
    assert 1 <= result["best_k"] <= X.shape[1]
    assert min(fits) == 32 and max(fits) == 48


def test_model_roundtrip_preserves_frozen_features(training_data):
    X, y, rows, names = training_data
    result = core.train(X, y, dict(scaler="none", k=2))
    config = core.FeatureConfig(128, 2, 60)
    package = pickle.loads(
        pickle.dumps(core.model_package(result["pipeline"], names, config, 128))
    )
    core.validate_package(package)
    restored = core.FeatureConfig(**package["config"])
    vector = core.prediction_features(
        rows[0], package["features"], restored, package["signal_length"]
    )
    assert package["pipeline"].predict(vector)[0] in [10, 30]
    assert restored == config
    with pytest.raises(ValueError, match="exactly"):
        core.prediction_features(np.ones(64), names, restored, 128)


def test_incompatible_and_incomplete_models_rejected(training_data):
    X, y, _, names = training_data
    result = core.train(X, y, {})
    package = core.model_package(result["pipeline"], names, core.FeatureConfig(), 128)
    with pytest.raises(ValueError):
        core.validate_package({"model": result["pipeline"]})
    package["sklearn"] = "0.0.0"
    with pytest.raises(ValueError, match="major/minor"):
        core.validate_package(package)


@pytest.mark.parametrize(
    "labels,fraction",
    [
        (np.zeros(10), 0.2),
        (np.array([1] * 9 + [2]), 0.2),
        (np.repeat([1, 2, 3], 2), 0.1),
    ],
)
def test_invalid_training_split(labels, fraction):
    with pytest.raises(ValueError):
        core.validate_training(np.ones((len(labels), 2)), labels, fraction)


def test_feature_schema_rejects_missing_and_duplicate_names():
    for names in [[], ["mean", "mean"], ["made_up"]]:
        with pytest.raises(ValueError):
            core.extract_features(np.ones((2, 16)), names, core.FeatureConfig())
