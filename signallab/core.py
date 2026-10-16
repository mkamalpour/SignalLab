"""Validated signal features and leakage-safe classification, independent of Tk."""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

TIME_FEATURES = (
    "mean",
    "median",
    "variance",
    "skewness",
    "kurtosis",
    "rms",
    "ptp",
    "iqr",
    "zero_crossings",
)
FREQUENCY_FEATURES = (
    "spec_centroid",
    "dom_freq",
    "spec_entropy",
    "fft_power",
    "band_power",
    "spectral_rolloff",
    "spectral_roughness",
)
FEATURE_NAMES = TIME_FEATURES + FREQUENCY_FEATURES


@dataclass(frozen=True)
class FeatureConfig:
    fs: float = 1024.0
    low: float = 0.0
    high: float = 512.0

    def __post_init__(self):
        if not np.isfinite([self.fs, self.low, self.high]).all() or self.fs <= 0:
            raise ValueError("Sampling rate must be finite and positive.")
        if not 0 <= self.low < self.high <= self.fs / 2:
            raise ValueError("Band must satisfy 0 <= low < high <= sampling rate / 2.")


def signal_matrix(values, sample_axis=None):
    a = np.asarray(values)
    if a.dtype.kind not in "biuf" or np.iscomplexobj(a):
        raise ValueError(
            "Choose a real numeric signal array, not a struct, string, or complex array."
        )
    a = np.squeeze(a)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    elif a.ndim == 2 and sample_axis == 0:
        a = a.T
    elif (
        a.ndim == 2
        and sample_axis is None
        and a.shape[0] == 1024
        and a.shape[1] != 1024
    ):
        a = a.T
    if a.ndim != 2 or not a.shape[0] or a.shape[1] < 4:
        raise ValueError(
            "Expected a vector or matrix with at least four samples per signal."
        )
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError(
            "Signals contain NaN or infinity; correct the source data first."
        )
    if np.abs(a).max() > 1e100:
        raise ValueError(
            "Signal magnitudes are too large for stable feature extraction."
        )
    return a


def read_mat(path):
    try:
        return {
            k: v
            for k, v in loadmat(path, squeeze_me=True, struct_as_record=False).items()
            if not k.startswith("__")
        }
    except NotImplementedError as e:
        raise ValueError(
            "MATLAB v7.3/HDF5 files are not supported. Save as MATLAB -v7 or -v6."
        ) from e


def labelled_mat(path, x_key="X", y_key="y"):
    data = read_mat(path)
    if x_key in data:
        if y_key not in data:
            raise ValueError(
                "Signal matrix found but labels are missing. Use the inspector for unlabelled data."
            )
        raw = np.asarray(data[x_key])
        y = np.asarray(data[y_key]).reshape(-1)
        if raw.ndim == 2 and raw.shape[0] != y.size and raw.shape[1] == y.size:
            raw = raw.T
        X = signal_matrix(raw, sample_axis=1)
        if len(X) != len(y):
            raise ValueError("The label count must equal the signal count.")
        if (
            y.dtype.kind not in "biuf"
            or not np.isfinite(y).all()
            or not np.equal(y, np.floor(y)).all()
        ):
            raise ValueError(
                "Labels must be finite integers; arbitrary integer class IDs are supported."
            )
        return X, y.astype(np.int64)
    import re

    keys = sorted(
        (k for k in data if re.fullmatch(r"[Xx]\d+", k)), key=lambda k: int(k[1:])
    )
    if not keys:
        raise ValueError(
            "Specify signal/label keys, or use separate X1, X2, ... class matrices."
        )
    matrices = [signal_matrix(data[k]) for k in keys]
    if len({a.shape[1] for a in matrices}) != 1:
        raise ValueError(
            "Class matrices have different signal lengths. No automatic padding or truncation is applied."
        )
    return np.vstack(matrices), np.concatenate(
        [np.full(len(a), int(k[1:]), dtype=int) for k, a in zip(keys, matrices)]
    )


def time_features(sig):
    s = signal_matrix(sig)[0]
    constant = np.ptp(s) == 0
    return dict(
        zip(
            TIME_FEATURES,
            (
                np.mean(s),
                np.median(s),
                np.var(s),
                0.0 if constant else skew(s),
                0.0 if constant else kurtosis(s),
                np.sqrt(np.mean(s * s)),
                np.ptp(s),
                np.percentile(s, 75) - np.percentile(s, 25),
                np.count_nonzero(np.signbit(s[:-1]) != np.signbit(s[1:])),
            ),
        )
    )


def spectrum(sig, fs):
    s = signal_matrix(sig)[0]
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sampling rate must be positive.")
    p = np.abs(rfft(s)) ** 2 / len(s) ** 2
    # One-sided mean-square contributions: DC/Nyquist occur once, interior bins twice.
    p[1 : -1 if len(s) % 2 == 0 else None] *= 2
    return rfftfreq(len(s), 1 / fs), p


def frequency_features(sig, config):
    f, p = spectrum(sig, config.fs)
    total = p.sum()
    probability = p / total if total else np.zeros_like(p)
    nonzero = probability[probability > 0]
    return dict(
        zip(
            FREQUENCY_FEATURES,
            (
                float(f @ probability),
                float(f[np.argmax(p)]) if total else 0.0,
                float(-np.sum(nonzero * np.log2(nonzero))),
                float(total),
                float(p[(f >= config.low) & (f <= config.high)].sum()),
                float(f[min(np.searchsorted(np.cumsum(probability), 0.85), len(f) - 1)])
                if total
                else 0.0,
                float(np.sum(np.diff(probability) ** 2)),
            ),
        )
    )


def extract_features(signals, names, config):
    names = tuple(names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(n not in FEATURE_NAMES for n in names)
    ):
        raise ValueError("Choose a nonempty, unique list of known features.")
    X = signal_matrix(signals, sample_axis=1)
    rows = []
    for s in X:
        values = {**time_features(s), **frequency_features(s, config)}
        rows.append([values[n] for n in names])
    result = np.asarray(rows)
    if not np.isfinite(result).all():
        raise ValueError("Features are not finite; check signal values.")
    return result


def validate_training(X, y, test_size):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if (
        X.ndim != 2
        or not X.shape[1]
        or y.ndim != 1
        or len(X) != len(y)
        or not np.isfinite(X).all()
    ):
        raise ValueError("Features and labels must be finite, nonempty and aligned.")
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or min(counts) < 2:
        raise ValueError(
            "Training needs at least two classes with two or more signals each."
        )
    if not 0 < test_size < 1:
        raise ValueError("Test fraction must be between zero and one.")
    n_test = int(np.ceil(len(X) * test_size))
    if min(n_test, len(X) - n_test) < len(counts):
        raise ValueError(
            "Train/test split is too small to include every class. Add samples or adjust test fraction."
        )
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


def safe_anova(X, y):
    # Constant features have no discriminating information; avoid NaN scores and noisy warnings.
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.filterwarnings(
            "ignore", message="Features .* are constant", category=UserWarning
        )
        scores, pvalues = f_classif(X, y)
    return np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max), np.nan_to_num(
        pvalues, nan=1.0
    )


def make_pipeline(
    model="svm", scaler="standard", k=0, C=1.0, kernel="rbf", class_weight="balanced"
):
    if model not in ("svm", "random_forest", "logistic") or scaler not in (
        "standard",
        "minmax",
        "none",
    ):
        raise ValueError("Unknown classifier or scaler.")
    if not np.isfinite(C) or C <= 0 or kernel not in ("rbf", "linear", "poly") or k < 0:
        raise ValueError("Invalid model parameters.")
    if class_weight not in ("balanced", "none"):
        raise ValueError("Unknown class weighting.")
    weight = None if class_weight == "none" else "balanced"
    estimator = (
        SVC(C=C, kernel=kernel, probability=True, class_weight=weight, random_state=42)
        if model == "svm"
        else RandomForestClassifier(
            n_estimators=100, class_weight=weight, random_state=42, n_jobs=1
        )
        if model == "random_forest"
        else LogisticRegression(
            C=C, max_iter=2000, class_weight=weight, random_state=42
        )
    )
    return Pipeline(
        [
            ("select", SelectKBest(safe_anova, k=k) if k else "passthrough"),
            (
                "scale",
                StandardScaler()
                if scaler == "standard"
                else MinMaxScaler()
                if scaler == "minmax"
                else "passthrough",
            ),
            ("classifier", estimator),
        ]
    )


def train(X, y, options):
    options = dict(options)
    test_size = options.pop("test_size", 0.2)
    grid = options.pop("grid", False)
    folds = options.pop("folds", 5)
    if options.get("k", 0) > X.shape[1]:
        raise ValueError("Selected feature count exceeds available features.")
    Xt, Xv, yt, yv = validate_training(X, y, test_size)
    pipeline = make_pipeline(**options)
    search_params = None
    if grid:
        minimum = min(np.unique(yt, return_counts=True)[1])
        if not 2 <= folds <= minimum:
            raise ValueError(
                f"CV folds must be between 2 and {minimum} for this training split."
            )
        name = options.get("model", "svm")
        params = (
            {"classifier__C": [0.1, 1, 10]}
            if name != "random_forest"
            else {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [None, 10],
            }
        )
        search = GridSearchCV(
            pipeline,
            params,
            cv=StratifiedKFold(folds, shuffle=True, random_state=42),
            n_jobs=1,
            error_score="raise",
        )
        search.fit(Xt, yt)
        pipeline = search.best_estimator_
        search_params = search.best_params_
    else:
        pipeline.fit(Xt, yt)
    predicted = pipeline.predict(Xv)
    return dict(
        pipeline=pipeline,
        accuracy=accuracy_score(yv, predicted),
        report=classification_report(yv, predicted, zero_division=0),
        cm=confusion_matrix(yv, predicted, labels=pipeline.classes_),
        classes=pipeline.classes_,
        train_count=len(yt),
        test_count=len(yv),
        best_params=search_params,
    )


def incremental_evaluation(X, y, test_size=0.2, folds=5):
    # Reserve the holdout BEFORE selecting k. Rank features separately inside each CV fold.
    Xt, Xv, yt, yv = validate_training(X, y, test_size)
    folds = min(folds, int(min(np.unique(yt, return_counts=True)[1])))
    if folds < 2:
        raise ValueError(
            "Incremental evaluation needs more training examples per class."
        )
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    scores = []
    for k in range(1, X.shape[1] + 1):
        scores.append(
            float(
                cross_val_score(
                    make_pipeline(k=k), Xt, yt, cv=cv, n_jobs=1, error_score="raise"
                ).mean()
            )
        )
    best_k = int(np.argmax(scores)) + 1
    pipeline = make_pipeline(k=best_k)
    pipeline.fit(Xt, yt)
    predicted = pipeline.predict(Xv)
    return dict(
        pipeline=pipeline,
        scores=scores,
        best_k=best_k,
        accuracy=accuracy_score(yv, predicted),
        report=classification_report(yv, predicted, zero_division=0),
        cm=confusion_matrix(yv, predicted, labels=pipeline.classes_),
        classes=pipeline.classes_,
        train_count=len(yt),
        test_count=len(yv),
    )


def prediction_features(signal, names, config, length):
    row = signal_matrix(signal)
    if row.shape != (1, length):
        raise ValueError(f"Model expects exactly {length} samples per signal.")
    return extract_features(row, names, config)


def model_package(pipeline, names, config, length):
    from dataclasses import asdict

    import sklearn

    return dict(
        format="signallab-v1",
        sklearn=sklearn.__version__,
        pipeline=pipeline,
        features=list(names),
        config=asdict(config),
        signal_length=length,
    )


def validate_package(package):
    import sklearn

    if not isinstance(package, dict) or package.get("format") != "signallab-v1":
        raise ValueError(
            "Unsupported model package. Retrain legacy models in SignalLab before exporting."
        )
    if package.get("sklearn", "").split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise ValueError(
            "Model scikit-learn major/minor version differs. Use the export environment or retrain."
        )
    pipe = package.get("pipeline")
    names = package.get("features")
    if (
        not isinstance(pipe, Pipeline)
        or not hasattr(pipe, "classes_")
        or not names
        or len(set(names)) != len(names)
        or any(n not in FEATURE_NAMES for n in names)
    ):
        raise ValueError("Model is unfitted or missing a valid feature schema.")
    if pipe.n_features_in_ != len(names):
        raise ValueError("Model feature count does not match its metadata.")
    FeatureConfig(**package["config"])
    if type(package.get("signal_length")) is not int or package["signal_length"] < 4:
        raise ValueError("Invalid model signal length.")
