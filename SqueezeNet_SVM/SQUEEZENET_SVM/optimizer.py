import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _svm_score(X, y, C, gamma, cv=5, seed=42):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr, yva = y[train_idx], y[val_idx]

        clf = make_pipeline(
            StandardScaler(),
            SVC(C=C, gamma=gamma, kernel="rbf", probability=True)
        )
        clf.fit(Xtr, ytr)
        scores.append(clf.score(Xva, yva))
    return float(np.mean(scores))


# Binary Q-BGWO for encoded parameters

def qbgwo_optimize(X, y, bits=10, cv=5, seed=42):
    rng = np.random.default_rng(seed)

    # Encode log10(C) in [-3, 3], log10(gamma) in [-5, 1]
    def decode(binary):
        b1 = binary[:bits]
        b2 = binary[bits:]
        v1 = int("".join(str(int(x)) for x in b1), 2) / (2**bits - 1)
        v2 = int("".join(str(int(x)) for x in b2), 2) / (2**bits - 1)
        logC = -3 + v1 * 6
        logG = -5 + v2 * 6
        return 10 ** logC, 10 ** logG

    n_wolves = 8
    iters = 12
    dim = bits * 2

    # Initialize quantum states (x,y) for each wolf
    x = np.full((n_wolves, dim), 0.5)
    yq = np.full((n_wolves, dim), 0.5)

    def binarize(xq, yq):
        # threshold with y^2
        return (xq >= (yq ** 2)).astype(int)

    alpha = beta = delta = None
    alpha_score = beta_score = delta_score = -1

    for _ in range(iters):
        # Evaluate wolves
        for i in range(n_wolves):
            binary = binarize(x[i], yq[i])
            C, gamma = decode(binary)
            score = _svm_score(X, y, C, gamma, cv=cv, seed=seed)

            if score > alpha_score:
                delta_score, delta = beta_score, beta
                beta_score, beta = alpha_score, alpha
                alpha_score, alpha = score, (x[i].copy(), yq[i].copy())
            elif score > beta_score:
                delta_score, delta = beta_score, beta
                beta_score, beta = score, (x[i].copy(), yq[i].copy())
            elif score > delta_score:
                delta_score, delta = score, (x[i].copy(), yq[i].copy())

        # Update (quantum-inspired)
        for i in range(n_wolves):
            r = rng.random(dim)
            x[i] = (alpha[0] + beta[0] + delta[0]) / 3.0
            yq[i] = (alpha[1] + beta[1] + delta[1]) / 3.0

            # small quantum rotation-like perturbation
            x[i] = np.clip(x[i] + 0.1 * (r - 0.5), 0.0, 1.0)
            yq[i] = np.clip(yq[i] + 0.1 * (r - 0.5), 0.0, 1.0)

    best_binary = binarize(alpha[0], alpha[1])
    best_C, best_gamma = decode(best_binary)
    return best_C, best_gamma, alpha_score
