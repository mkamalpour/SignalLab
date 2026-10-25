import numpy as np
import pytest
from scipy.io import savemat
from signallab import core


@pytest.mark.parametrize('labels', [np.array([2.**63, 0]), np.array([2**64 - 1, 0], dtype=np.uint64)])
def test_out_of_range_labels_do_not_silently_change_class(tmp_path, labels):
    path = tmp_path / 'labels.mat'
    savemat(path, {'X': np.ones((2, 8)), 'y': labels})
    with pytest.raises(ValueError, match='64-bit'):
        core.labelled_mat(path)


def test_duplicate_numeric_class_aliases_are_rejected(tmp_path):
    path = tmp_path / 'aliases.mat'
    savemat(path, {'X1': np.ones((2, 8)), 'x01': np.zeros((2, 8))})
    with pytest.raises(ValueError, match='unique'):
        core.labelled_mat(path)


def test_nonstring_package_version_has_useful_error():
    with pytest.raises(ValueError, match='version'):
        core.validate_package({'format': 'signallab-v1', 'sklearn': None})
