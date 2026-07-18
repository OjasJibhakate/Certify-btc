"""
conformal.py — RAPS conformal prediction (Phase 6).

A single predicted class hides risk. Conformal prediction instead returns a SET of classes with
a guarantee: after calibrating a threshold, the set contains the true class at least
(1 - alpha) of the time (we target 95% coverage). Easy scans -> a 1-class set; ambiguous scans
-> a bigger set — and that bigger set is itself a signal ("this one is hard, look closer").

RAPS (Angelopoulos et al. 2021) = Adaptive Prediction Sets + a size penalty so the sets don't
blow up. We use SPLIT conformal: calibrate the threshold on a held-out labelled set (we use the
validation set), then apply it to new data. The guarantee is distribution-free — it holds no
matter how (in)accurate the model is, as long as calibration and test data look alike.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config


class ConformalRAPS:
    def __init__(self, coverage=0.95, lam=0.01, k_reg=1, randomized=True, seed=0):
        self.alpha = 1.0 - coverage       # miscoverage we tolerate
        self.lam = lam                     # size penalty strength
        self.k_reg = k_reg                 # sets up to this size are penalty-free
        self.randomized = randomized       # randomization gives exact (not conservative) coverage
        self.rng = np.random.default_rng(seed)
        self.tau = None

    def _true_class_scores(self, probs, labels):
        """RAPS nonconformity score of the TRUE class for each calibration sample."""
        n, K = probs.shape
        order = np.argsort(-probs, axis=1)                 # class ids sorted by prob desc
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        scores = np.empty(n)
        for i in range(n):
            rank = int(np.where(order[i] == labels[i])[0][0])   # 0-indexed rank of true class
            cumulative = cum[i, rank]                            # includes the true class prob
            reg = self.lam * max(0, (rank + 1) - self.k_reg)     # penalty for deep-ranked truth
            if self.randomized:
                cumulative -= self.rng.random() * sorted_p[i, rank]
            scores[i] = cumulative + reg
        return scores

    def calibrate(self, probs, labels):
        probs, labels = np.asarray(probs), np.asarray(labels)
        scores = self._true_class_scores(probs, labels)
        n = len(scores)
        # The finite-sample-valid quantile level for split conformal.
        q = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.tau = float(np.quantile(scores, q, method="higher"))
        return self.tau

    def predict(self, probs):
        """Return a prediction SET (list of class ids) for each row of probs."""
        if self.tau is None:
            raise RuntimeError("Call calibrate() before predict().")
        probs = np.asarray(probs)
        n, K = probs.shape
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        sets = []
        for i in range(n):
            cum, chosen = 0.0, []
            for k in range(K):
                cum += sorted_p[i, k]
                reg = self.lam * max(0, (k + 1) - self.k_reg)
                score = cum + reg
                if self.randomized:
                    score -= self.rng.random() * sorted_p[i, k]
                chosen.append(int(order[i, k]))
                if score > self.tau:               # stop once we've crossed the threshold
                    break
            sets.append(sorted(chosen))
        return sets


def coverage_and_size(sets, labels):
    """Empirical coverage (fraction of sets containing the true label) and mean set size."""
    labels = np.asarray(labels)
    covered = sum(1 for s, y in zip(sets, labels) if int(y) in s)
    avg_size = float(np.mean([len(s) for s in sets]))
    return covered / len(labels), avg_size


class MondrianConformalRAPS:
    """Class-CONDITIONAL RAPS. Standard conformal guarantees coverage *overall*; a hard class
    (glioma) can still be under-covered. Mondrian calibrates a SEPARATE threshold per class,
    so each class — including glioma — gets its own >= (1-alpha) coverage guarantee. The price
    is somewhat larger sets for the hard classes."""

    def __init__(self, coverage=0.95, lam=0.01, k_reg=1, randomized=True, seed=0):
        self.alpha = 1.0 - coverage
        self.lam = lam
        self.k_reg = k_reg
        self.randomized = randomized
        self.rng = np.random.default_rng(seed)
        self.tau = None
        self.num_classes = None

    def _score(self, prob_row, label, u=None):
        """RAPS nonconformity score of `label` for one probability row."""
        order = np.argsort(-prob_row)
        rank = int(np.where(order == label)[0][0])        # 0-indexed rank of `label`
        cumulative = prob_row[order][:rank + 1].sum()      # includes `label`'s prob
        reg = self.lam * max(0, (rank + 1) - self.k_reg)
        if self.randomized:
            u = self.rng.random() if u is None else u
            cumulative -= u * prob_row[order][rank]
        return cumulative + reg

    def calibrate(self, probs, labels):
        probs, labels = np.asarray(probs), np.asarray(labels)
        self.num_classes = probs.shape[1]
        self.tau = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            idx = np.where(labels == c)[0]                 # calibrate class c on its OWN samples
            scores = np.array([self._score(probs[i], c) for i in idx])
            n = len(scores)
            q = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.tau[c] = float(np.quantile(scores, q, method="higher"))
        return self.tau

    def predict(self, probs):
        probs = np.asarray(probs)
        sets = []
        for i in range(len(probs)):
            s = [k for k in range(self.num_classes) if self._score(probs[i], k) <= self.tau[k]]
            sets.append(sorted(s) if s else [int(probs[i].argmax())])  # never emit an empty set
        return sets


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from modules.model import CertifyBTC
    from modules.datasets import build_dataloaders
    from train import load_checkpoint

    device = config.DEVICE
    model = CertifyBTC().to(device)
    model.eval()
    ck = os.path.join(config.CHECKPOINT_DIR, "stage1_best.pth")
    if os.path.exists(ck):
        load_checkpoint(ck, model, device=device)

    _, val_loader, test_loader = build_dataloaders()

    @torch.no_grad()
    def get_probs(loader):
        P, Y = [], []
        for x, y in loader:
            P.append(F.softmax(model(x.to(device)), dim=1).cpu().numpy())
            Y.append(y.numpy())
        return np.concatenate(P), np.concatenate(Y)

    cal_p, cal_y = get_probs(val_loader)     # calibrate on validation
    test_p, test_y = get_probs(test_loader)  # evaluate on test

    cp = ConformalRAPS(coverage=config.CONFORMAL_COVERAGE)
    tau = cp.calibrate(cal_p, cal_y)
    sets = cp.predict(test_p)
    cov, size = coverage_and_size(sets, test_y)

    print(f"  target coverage    : {config.CONFORMAL_COVERAGE:.2f}  (tau={tau:.3f})")
    print(f"  empirical coverage : {cov:.3f}   avg set size : {size:.2f}")
    print(f"  (small local sets = noisy; real calibration uses the full data)")
    print("conformal OK.")
