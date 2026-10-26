# Engineering audit — 2026-10-26

Purpose: Signal features, supervised classification and MATLAB inspection.

Classification: **A** (single application/tool).
Final structure: signallab/ for core and desktop tools; tests/. Existing capabilities and source material were retained.

| Assessment | Result |
| --- | --- |
| Idea strength | 7/10 |
| Implementation strength | 8/10 |
| Core strength | Numerical invariants, preprocessing boundaries and saved pipeline metadata. |
| Main limitation | Real signal evaluation depends on verified sample rates, labels and representative recordings. |
| Readiness | Local prototype/tool with the verified baseline below; no production certification |

Scores are subjective engineering estimates, not measured product or model quality.

## Delivered

Fixed process-local discovery of the active environment's bundled Arrow runtime and native BLAS directory. Added test.py to initialize runtime before pytest and exclude unrelated global plugins. No installed environment files were edited.

Assigned environment: **ml**. Run `.\run.cmd --check`, then `.\run.cmd`.
The launcher installs nothing and preserves its child exit status. On this workstation
it selects the assigned interpreter; standalone clones fall back to active Python.
VS Code's local interpreter settings were aligned with the same assignment.

## Verification

- **Automated tests:** 48 passed; 0 reported skips; 0 private-report cases deselected.
- **Startup check:** `run.cmd --check` passed from outside the repository;
  required startup modules were imported in the assigned environment.
- **Tested runtime:** Windows; Conda ml; Python 3.10.19.
- **Data handling:** tests used temporary/synthetic fixtures. No live transactions,
  provider generation, deployment, new model downloads or package installs were performed.
- **Repository check:** origin/main fetched before publication; no pre-existing local
  changes were present. A tracked-file scan found no matches for the high-confidence
  credential patterns checked. That scan is not a guarantee that all personal data
  or arbitrary secret formats are absent. Ignored local data was not staged.
- **Limits:** optional external workflows, representative real-data quality and all
  possible platform/version combinations remain unverified.

Test command from the repository root:

```powershell
conda activate ml
python test.py
```

## Next three upgrades

1. Import schema presets.
2. Cross-session evaluation reports.
3. Feature provenance in comparison exports.

Expected benefit: clearer decisions, more reproducible workflows and stronger evidence
for quality. These are proposed upgrades, not implemented features or measured speedups.

External assets: No model downloads are needed; real validated signal data is needed for domain claims.

No root license file is set. No license was selected or changed.

## Additional implementation pass

Balanced accuracy, macro F1 and a training-majority-class baseline accompany held-out classification results.

Use: Train a classifier; inspect the training report for balanced metrics and the majority baseline.

Verification: the complete offline suite passed in the assigned **ml**
environment after this change (48 tests). Regression fixtures remain
temporary/synthetic. Optional real-model/data quality and external integrations
remain unverified. No packages, models or additional environments were installed.

The earlier numeric scores are subjective historical assessments; new features do
not automatically establish a higher score or production readiness.
