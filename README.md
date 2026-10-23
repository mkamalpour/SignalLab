# SignalLab

A desktop workbench for exploring MATLAB signal datasets and building reproducible signal classifiers. Inspect waveforms and spectra, compare features, evaluate models, and export complete prediction packages through a six-tab Tkinter interface.

## Two tools, one launcher

- **Analysis workbench**: synthetic data, MATLAB class matrices and CSV class folders; signal previews; 16 time/spectral features; boxplots and standardized PCA; SVM, random forest and logistic regression; optional ANOVA selection, scaling and grid search; incremental feature-count evaluation; prediction probes; CSV/model export.
- **MATLAB inspector**: browse variable names, shapes and types; select row/column orientation and signal index; set sampling rate; inspect time traces, mean-square FFT power and statistics; open histogram/spectrogram reports; zoom and save figures.

## Run locally

Use **Python 3.13** for the tested environment. The pinned scientific stack also supports Python 3.10-3.12; those interpreters have not been separately verified here. Python installations must include Tkinter (the standard Windows installer does; Linux may need its distribution's `python3-tk` package).

```powershell
conda activate base  # Or activate another existing compatible environment.
# Optional dependency setup; requires explicit permission and connectivity.
python -m pip install -r requirements.txt
python main.py
python main.py --inspect
```

Use `python` from your active compatible environment on other platforms too.

Start with **Synthetic**, click **Load / Generate**, extract features, then train. Synthetic generation uses seed 42. Drag the horizontal divider to resize the workflow and plot workspace. The plot toolbar supports pan, zoom and image export.

## Input formats

- **MATLAB labelled matrix**: `X` is signals by samples; `y` is an equal-length vector of integer class IDs. Custom variable names are supported. A sample-by-signals matrix is transposed when the label count establishes orientation.
- **MATLAB class matrices**: `X1`, `X2`, ... each contain one class. Rows are signals; matrices with 1024 rows and a different column count are recognized as transposed 1024-sample signals. The numeric suffix is the class ID. Use the inspector's explicit orientation control for other layouts.
- **CSV folders**: one subfolder per class, one headerless numeric row or column per `.csv` file. All signals must have equal length. The activity log records folder-to-class IDs.
- **Unlabelled data**: use the MATLAB inspector. The classifier never invents class labels, pads or truncates signals silently. Scalars, structs, complex arrays, NaN and infinity are rejected with an explanation.
- MATLAB v7.3/HDF5 is not supported by the loader. Resave the required variables using MATLAB `save(..., '-v7')`.

Your local `dataset.mat` stays outside Git, along with exported features, models and caches. A dataset is not needed to try the synthetic demonstration.

## Scientific behavior

Enter the **actual sampling rate in Hz** before extracting features. The 1024 Hz default belongs to the demonstration and is not inferred from your MATLAB file. Band limits must lie between zero and Nyquist.

One-sided FFT bin contributions are normalized to mean-square amplitude: their total equals RMS squared, including the correct DC/Nyquist treatment. Band power sums bins in the chosen frequency interval. `spectral_roughness` describes adjacent-bin differences; it is not temporal spectral flux. Constant signals have zero skewness and excess kurtosis by convention. PCA is exploratory, uses standardized features, and does not determine the model score.

The train/test split happens before fitting selection/scaling. Grid search fits the entire pipeline inside each fold. Incremental evaluation chooses the feature count using training cross-validation and evaluates the final model on the reserved holdout. Its graph shows training CV scores, while the report shows holdout accuracy. Repeating experiments against that same holdout can still overfit it.

This is **independent-signal classification**, not blind source separation. Correlated windows, repeated subjects or sessions require group-aware evaluation outside this app. Synthetic results do not establish real-data accuracy. Dataset prediction previews may include training examples; model probability estimates are not calibrated confidence guarantees.

Reference behavior: [scikit-learn leakage prevention](https://scikit-learn.org/stable/common_pitfalls) and [SciPy MATLAB loading](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html).

## Model compatibility

Exports contain the fitted pipeline, feature order, signal length, sampling rate, band limits and scikit-learn version. Predictions use that saved configuration even if current UI settings change. Unsupported legacy packages and different scikit-learn major/minor versions are rejected instead of guessing a feature layout. Retrain legacy models to export the current format.

**Only import trusted pickle files:** loading pickle can execute code. Version checks do not make an untrusted file safe.

## Validation and structure

```powershell
# Optional dependency setup; requires explicit permission and connectivity.
python -m pip install pytest
python -m pytest tests -q
```

The regression suite checks FFT power conservation, known-frequency signals, constant/invalid inputs, MATLAB orientation and labels, all classifier/scaler combinations, fold-local preprocessing, reserved-holdout feature selection, and model round-trips. Desktop workflows are also checked separately against synthetic data and the local MATLAB dataset.

```text
main.py                 # launch either desktop tool
requirements.txt        # one tested dependency set
signallab/
  analyzer.py           # six-tab desktop workbench
  inspector.py          # dedicated MATLAB browser and reports
  core.py               # GUI-independent numerical and ML logic
tests/                  # focused scientific regression checks
```

No real dataset or trained model is included in this repository.

Current lightweight check (Windows, existing Conda base / Python 3.13.5): **38 passed**. No dependency installs, model downloads, live publishing, or heavy datasets were used.
