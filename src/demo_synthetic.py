"""Synthetic demonstration of frame-level leakage (no real data needed).

Each simulated subject has a strong individual "fingerprint" and only a weak
diagnostic signal, and contributes many frames. A classifier evaluated with an
image-level split can memorise fingerprints and look far better than it is;
a subject-level split shows the honest accuracy.

Run:  python src/demo_synthetic.py
"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from splits import check_leakage, overlap_probability, subject_level_split

RNG = np.random.default_rng(0)
N_SUBJECTS_PER_CLASS = 30
N_CLASSES = 3
FRAMES_PER_SUBJECT = 60
N_FEATURES = 200
CLASS_SIGNAL = 0.15     # weak true diagnostic effect
SUBJECT_SIGNAL = 1.0    # strong per-subject fingerprint
NOISE = 1.0


def make_data():
    class_means = RNG.normal(0, CLASS_SIGNAL, size=(N_CLASSES, N_FEATURES))
    X, y, subj = [], [], []
    sid = 0
    for c in range(N_CLASSES):
        for _ in range(N_SUBJECTS_PER_CLASS):
            fingerprint = RNG.normal(0, SUBJECT_SIGNAL, size=N_FEATURES)
            frames = class_means[c] + fingerprint + RNG.normal(0, NOISE, size=(FRAMES_PER_SUBJECT, N_FEATURES))
            X.append(frames)
            y += [c] * FRAMES_PER_SUBJECT
            subj += [sid] * FRAMES_PER_SUBJECT
            sid += 1
    return np.vstack(X), np.array(y), np.array(subj)


def evaluate(X, y, train_idx, test_idx):
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[train_idx], y[train_idx])
    return clf.score(X[test_idx], y[test_idx])


def main():
    X, y, subj = make_data()
    idx = np.arange(len(y))

    tr, te = train_test_split(idx, test_size=0.2, random_state=0, stratify=y)
    leaked = check_leakage(subj[tr], subj[te], raise_error=False)
    acc_image = evaluate(X, y, tr, te)

    tr_s, te_s = subject_level_split(subj, test_size=0.2, seed=0)
    acc_subject = evaluate(X, y, tr_s, te_s)

    n_subj = len(np.unique(subj))
    print(f"Subjects: {n_subj}, frames per subject: {FRAMES_PER_SUBJECT}, classes: {N_CLASSES}")
    print(f"Image-level split  : accuracy {acc_image:.3f}  "
          f"(subjects on both sides: {len(leaked)}/{n_subj})")
    print(f"Subject-level split: accuracy {acc_subject:.3f}  (subjects on both sides: 0)")
    print(f"Chance level       : {1 / N_CLASSES:.3f}")
    print(f"P(subject on both sides | {FRAMES_PER_SUBJECT} frames, 20% test) = "
          f"{overlap_probability(FRAMES_PER_SUBJECT):.6f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(here, "..", "figures", "synthetic_demo.png")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        bars = ax.bar(["Image-level\nsplit", "Subject-level\nsplit"], [acc_image, acc_subject],
                      color=["#c0504d", "#1f5f8b"], width=0.55)
        ax.axhline(1 / N_CLASSES, ls="--", c="gray", lw=1)
        ax.text(1.32, 1 / N_CLASSES + 0.01, "chance", color="gray", fontsize=8)
        for b, v in zip(bars, [acc_image, acc_subject]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Test accuracy")
        ax.set_title("Same data, same model, different split (synthetic)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"Saved {os.path.normpath(out)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
