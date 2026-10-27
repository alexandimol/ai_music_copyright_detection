# -*- coding: utf-8 -*-
"""
evaluate.py
===========
Συγκριτική αξιολόγηση όλων των μεθόδων και παραγωγή των πινάκων και
γραφημάτων του κεφαλαίου «Αποτελέσματα».

Διαβάζει τις βαθμολογίες που αποθήκευσαν τα baselines.py και train.py
(outputs/results/scores_<μέθοδος>_<σύνολο>.npy) — όποιες μέθοδοι υπάρχουν.

Μετρικές (όλες ανεξάρτητες από το decision threshold):
    ROC-AUC       : πιθανότητα ένα τυχαίο ζεύγος αντιγραφής να βαθμολογηθεί
                    υψηλότερα από ένα τυχαίο ανεξάρτητο ζεύγος (Fawcett, 2006).
    AP            : μέση ακρίβεια (average precision), η περιοχή κάτω από την
                    καμπύλη ακρίβειας–ανάκλησης.
    TPR@1%FPR     : ποσοστό αντιγραφών που εντοπίζονται όταν επιτρέπουμε μόλις
                    1% ψευδείς ενδείξεις σε ανεξάρτητα έργα. Σημαντικό σε νομικό
                    πλαίσιο, όπου μια ψευδής κατηγορία έχει υψηλό κόστος.

Εκτέλεση:
    python evaluate.py
"""

import csv

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import config
import pairs as PR
import plot_style as S

plt = S.plt


# ===========================================================================
# 1. Φόρτωση βαθμολογιών
# ===========================================================================
def load_scores(split):
    """Επιστρέφει {μέθοδος: βαθμολογίες} για όσες μεθόδους υπάρχουν αποθηκευμένες."""
    out = {}
    for name in S.METHOD_ORDER:
        path = config.RESULTS_DIR / f"scores_{name}_{split}.npy"
        if path.exists():
            out[name] = np.load(path).ravel()
    return out


def tpr_at_fpr(y, s, fpr_max=0.01):
    """Μέγιστο TPR για FPR ≤ fpr_max."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def print_table(title, header, rows):
    """Εκτύπωση πίνακα στην κονσόλα."""
    print(f"\n{title}")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for row in [header] + rows:
        print("  " + "  ".join(str(x).ljust(w) for x, w in zip(row, widths)))


# ===========================================================================
# 2. Πίνακες
# ===========================================================================
def check_lengths(data, scores):
    """Αφαιρεί μεθόδους με λάθος πλήθος βαθμολογιών (π.χ. από τρέξιμο --quick)."""
    for m in list(scores):
        if len(scores[m]) != len(data["label"]):
            print(f"Προσοχή: παράλειψη {m} ({len(scores[m])} βαθμολογίες αντί για "
                  f"{len(data['label'])} — ξανατρέξτε τη μέθοδο χωρίς --quick)")
            del scores[m]
    return scores


def main_table(data, scores):
    y = data["label"]
    rows = [[S.METHOD_LABELS[m], f"{roc_auc_score(y, s):.3f}",
             f"{average_precision_score(y, s):.3f}", f"{tpr_at_fpr(y, s):.3f}"]
            for m, s in scores.items()]
    header = ["Μέθοδος", "ROC-AUC", "AP", "TPR@1%FPR"]
    print_table("Πίνακας 1 — Συνολική απόδοση (test set)", header, rows)
    write_csv(config.RESULTS_DIR / "table_overall.csv", header, rows)


def per_type_tables(data, scores):
    y, types = data["label"], data["type"]
    # (α) κάθε τύπος αντιγραφής έναντι όλων των ανεξάρτητων ζευγών
    header = ["Μέθοδος"] + [S.TYPE_LABELS[k] for k in PR.POSITIVE_TYPES]
    rows = []
    for m, s in scores.items():
        row = [S.METHOD_LABELS[m]]
        for k in PR.POSITIVE_TYPES:
            mask = (types == k) | (y == 0)
            row.append(f"{roc_auc_score(y[mask], s[mask]):.3f}")
        rows.append(row)
    print_table("Πίνακας 2 — ROC-AUC ανά τύπο αντιγραφής", header, rows)
    write_csv(config.RESULTS_DIR / "table_positive_types.csv", header, rows)

    # (β) όλες οι αντιγραφές έναντι κάθε τύπου ανεξάρτητου ζεύγους
    header = ["Μέθοδος"] + [S.TYPE_LABELS[k] for k in PR.NEGATIVE_TYPES]
    rows = []
    for m, s in scores.items():
        row = [S.METHOD_LABELS[m]]
        for k in PR.NEGATIVE_TYPES:
            mask = (types == k) | (y == 1)
            row.append(f"{roc_auc_score(y[mask], s[mask]):.3f}")
        rows.append(row)
    print_table("Πίνακας 3 — ROC-AUC έναντι κάθε τύπου ανεξάρτητου ζεύγους", header, rows)
    write_csv(config.RESULTS_DIR / "table_negative_types.csv", header, rows)


def remelodized_table(data, scores):
    """
    Πίνακας 4 (έκδοση v2): πόσο καλά ξεχωρίζει κάθε μέθοδος τις πραγματικές
    αντιγραφές από τμήματα με ΙΔΙΟ ρυθμό αλλά νέα μελωδία (70–100% αλλαγμένα
    τονικά ύψη). Υψηλή τιμή = η μέθοδος δεν «ξεγελιέται» από τον ρυθμό.
    """
    if not (config.PAIRS_DIR / "test_remelodized.npz").exists():
        return
    extra = load_scores("test_remelodized")
    n_extra = len(PR.load_pairs("test_remelodized")["label"])
    pos = data["label"] == 1
    rows = []
    for m, s in scores.items():
        if m not in extra or len(extra[m]) != n_extra:
            continue
        y = np.concatenate([np.ones(pos.sum()), np.zeros(n_extra)])
        sc = np.concatenate([s[pos], extra[m]])
        # ποσοστό των «νέων μελωδιών» που βαθμολογούνται ψηλότερα από το
        # decision threshold στο οποίο η μέθοδος έχει 1% ψευδείς ενδείξεις στα κανονικά αρνητικά
        thr = np.quantile(s[data["label"] == 0], 0.99)
        rows.append([S.METHOD_LABELS[m], f"{roc_auc_score(y, sc):.3f}",
                     f"{100 * (extra[m] > thr).mean():.1f}%"])
    header = ["Μέθοδος", "ROC-AUC (αντιγραφές έναντι νέας μελωδίας)",
              "Ψευδείς ενδείξεις στο όριο 1%FPR"]
    print_table("Πίνακας 4 — Νέα μελωδία με ίδιο ρυθμό (έλεγχος «συντόμευσης»)",
                header, rows)
    write_csv(config.RESULTS_DIR / "table_remelodized.csv", header, rows)


# ===========================================================================
# 3. Γραφήματα
# ===========================================================================
def plot_roc(data, scores, path):
    """Καμπύλες ROC όλων των μεθόδων (λογαριθμικός άξονας FPR για τις χαμηλές τιμές)."""
    y = data["label"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for m, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s)
        st = S.method_style(m)
        ax.plot(np.maximum(fpr, 1e-4), tpr, color=st["color"], linestyle=st["linestyle"],
                label=f"{S.METHOD_LABELS[m]} (AUC {roc_auc_score(y, s):.3f})")
    ax.axvline(0.01, color=S.MUTED, linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_xlabel("Ποσοστό ψευδών ενδείξεων (FPR, λογαριθμική κλίμακα)")
    ax.set_ylabel("Ποσοστό εντοπισμένων αντιγραφών (TPR)")
    ax.set_title("Καμπύλες ROC (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def percentile_vs_negatives(values, negatives):
    """
    Εκφράζει μια βαθμολογία ως ποσοστό των ανεξάρτητων ζευγών (test set) που
    έχουν ΜΙΚΡΟΤΕΡΗ βαθμολογία. Έτσι μέθοδοι με διαφορετικές κλίμακες
    βαθμολογιών γίνονται συγκρίσιμες (0 = σαν ανεξάρτητο έργο, 1 = πιο όμοιο
    από κάθε ανεξάρτητο ζεύγος).
    """
    neg = np.sort(negatives)
    return np.searchsorted(neg, values, side="left") / len(neg)


def plot_sweep(sweep_name, xlabel, x_transform, test_data, test_scores, path):
    """Μέση κανονικοποιημένη βαθμολογία ως συνάρτηση του βαθμού αντιγραφής."""
    data = PR.load_pairs(sweep_name)
    scores = load_scores(sweep_name)
    degree = np.round(data["degree"], 3)
    levels = np.unique(degree)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rows = []
    for m, s in scores.items():
        if m not in test_scores:
            continue
        if len(s) != len(degree):
            print(f"  Προσοχή: παράλειψη {m} ({len(s)} βαθμολογίες αντί για "
                  f"{len(degree)} — ξανατρέξτε τη μέθοδο χωρίς --quick)")
            continue
        pct = percentile_vs_negatives(s, test_scores[m][test_data["label"] == 0])
        means = np.array([pct[degree == lv].mean() for lv in levels])
        st = S.method_style(m)
        ax.plot(x_transform(levels), means, color=st["color"], marker=st["marker"],
                linestyle=st["linestyle"], label=S.METHOD_LABELS[m])
        rows.append([S.METHOD_LABELS[m]] + [f"{v:.3f}" for v in means])
    ax.axhline(0.99, color=S.MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Εκατοστημόριο ως προς τα ανεξάρτητα ζεύγη")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(path)
    plt.close(fig)
    header = ["Μέθοδος"] + [f"{v:g}" for v in x_transform(levels)]
    write_csv(config.RESULTS_DIR / f"table_{sweep_name}.csv", header, rows)
    print_table(f"Καμπύλη {sweep_name} (μέσο εκατοστημόριο)", header, rows)


def main():
    config.ensure_dirs()
    data = PR.load_pairs("test")
    scores = check_lengths(data, load_scores("test"))
    if not scores:
        raise FileNotFoundError("Δεν βρέθηκαν βαθμολογίες. Τρέξτε πρώτα baselines.py / train.py")
    print("Μέθοδοι:", ", ".join(S.METHOD_LABELS[m] for m in scores))

    main_table(data, scores)
    per_type_tables(data, scores)
    remelodized_table(data, scores)
    plot_roc(data, scores, config.FIGURES_DIR / "roc_test.png")
    plot_sweep("sweep_melodic", "Ποσοστό αλλαγμένων νοτών (%)",
               lambda d: np.round(100 * (1 - d)).astype(int), data, scores,
               config.FIGURES_DIR / "sweep_melodic.png")
    plot_sweep("sweep_partial", "Αντιγραμμένα μέτρα (από 8)",
               lambda d: np.round(d * config.SEGMENT_BEATS / 4).astype(int), data, scores,
               config.FIGURES_DIR / "sweep_partial.png")
    print(f"\nΠίνακες: {config.RESULTS_DIR}\nΓραφήματα: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
