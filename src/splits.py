"""Subject-level splitting and leakage checks for neuroimaging datasets.

When every subject contributes many images (fMRI frames, slices, augmented
copies), the images must be split by *subject*, not by image. Otherwise the
same person appears in both training and test data and accuracy is inflated.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def subject_level_split(subject_ids, test_size=0.2, seed=0):
    """Return (train_idx, test_idx) so that no subject is on both sides."""
    subject_ids = np.asarray(subject_ids)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(subject_ids)), groups=subject_ids))
    check_leakage(subject_ids[train_idx], subject_ids[test_idx])
    return train_idx, test_idx


def subject_level_kfold(labels, subject_ids, n_splits=5, seed=0):
    """Yield (train_idx, test_idx) for stratified, subject-grouped K-fold CV."""
    labels, subject_ids = np.asarray(labels), np.asarray(subject_ids)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(np.zeros(len(labels)), labels, groups=subject_ids):
        check_leakage(subject_ids[train_idx], subject_ids[test_idx])
        yield train_idx, test_idx


def check_leakage(train_subjects, test_subjects, raise_error=True):
    """Report subjects that appear in both partitions.

    Returns the set of overlapping subject IDs. Raises ValueError if any are
    found and ``raise_error`` is True.
    """
    overlap = set(np.asarray(train_subjects).tolist()) & set(np.asarray(test_subjects).tolist())
    if overlap and raise_error:
        raise ValueError(
            f"Data leakage: {len(overlap)} subject(s) appear in both train and test, "
            f"e.g. {sorted(overlap)[:5]}"
        )
    return overlap


def overlap_probability(n_frames, test_fraction=0.2):
    """Probability that a subject with ``n_frames`` images lands on BOTH sides
    of a random image-level split: 1 - (1-p)^n - p^n."""
    p = test_fraction
    return 1.0 - (1.0 - p) ** n_frames - p ** n_frames
