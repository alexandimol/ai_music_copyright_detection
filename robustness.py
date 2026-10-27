# -*- coding: utf-8 -*-
"""
robustness.py
=============
Εκτίμηση της αβεβαιότητας των αποτελεσμάτων, από δύο πηγές:

  (Α) Στοχαστικότητα της εκπαίδευσης — επαναλήψεις με διαφορετικό seed.
      Για κάθε Siamese μοντέλο (v1, v2, πλήρες και μόνο μελωδία) συγκεντρώνει
      την αρχική εκτέλεση (seed 42, π.χ. siamese_v2) και όσες επαναλήψεις
      υπάρχουν (π.χ. siamese_v2_s1, _s2, _s3, από «train.py --seed N») και
      αναφέρει μέσο όρο ± τυπική απόκλιση κάθε μετρικής.

  (Β) Δειγματοληπτική αβεβαιότητα του συνόλου ελέγχου — bootstrap.
      Για κάθε μέθοδο (Siamese και κλασικές), διαστήματα εμπιστοσύνης 95%
      με τη μέθοδο των εκατοστημορίων (Efron and Tibshirani, 1993).
      Η επαναδειγματοληψία γίνεται σε επίπεδο ΤΡΑΓΟΥΔΙΟΥ (cluster bootstrap):
      επιλέγονται με επανάθεση τραγούδια του συνόλου ελέγχου και κρατούνται
      όλα τα ζεύγη τους. Έτσι λαμβάνεται υπόψη ότι τα ζεύγη του ίδιου
      τραγουδιού δεν είναι ανεξάρτητα μεταξύ τους.

Μετρικές (ίδιες με τους Πίνακες 1, 2 και 4 του evaluate.py):
    auc, ap, tpr1      : ROC-AUC, AP, TPR@1%FPR στο σύνολο ελέγχου
    auc_partial        : ROC-AUC μερικής αντιγραφής έναντι όλων των αρνητικών
    fp_rem_05          : ψευδείς ενδείξεις (p > 0.5) σε «νέα μελωδία, ίδιος ρυθμός»
                         (μόνο για τα Siamese, που δίνουν βαθμολογία στο [0, 1])
    auc_rem, fp_rem_1  : Πίνακας 4 — AUC αντιγραφών έναντι νέας μελωδίας και
                         ψευδείς ενδείξεις στο όριο του 1% FPR

Εκτέλεση (αφού τρέξουν οι επαναλήψεις του train.py, βλ. README):
    python robustness.py
    python robustness.py --n_boot 2000
"""

import argparse
import csv
import json
import re

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import config
import pairs as PR
import plot_style as S

SIAMESE = ["siamese_v2", "siamese_melody_v2", "siamese", "siamese_melody"]
METRICS = ["auc", "ap", "tpr1", "auc_partial", "fp_rem_05", "auc_rem", "fp_rem_1"]
METRIC_LABELS = {
    "auc": "ROC-AUC", "ap": "AP", "tpr1": "TPR@1%FPR",
    "auc_partial": "AUC μερικής αντιγραφής",
    "fp_rem_05": "Ψευδείς ενδ. νέας μελωδίας (p>0.5)",
    "auc_rem": "AUC έναντι νέας μελωδίας",
    "fp_rem_1": "Ψευδείς ενδ. νέας μελωδίας (όριο 1%FPR)",
}
PERCENT = {"fp_rem_05", "fp_rem_1"}   # μετρικές που εμφανίζονται ως ποσοστό


# ===========================================================================
# 1. Βοηθητικές συναρτήσεις
# ===========================================================================
def tpr_at_fpr(y, s, fpr_max=0.01):
    """Μέγιστο TPR για FPR ≤ fpr_max (όπως στο evaluate.py)."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


def load(name, split, n):
    """Φορτώνει βαθμολογίες· None αν λείπουν ή έχουν λάθος πλήθος (π.χ. --quick)."""
    path = config.RESULTS_DIR / f"scores_{name}_{split}.npy"
    if not path.exists():
        return None
    s = np.load(path).ravel()
    if len(s) != n:
        print(f"  Προσοχή: παράλειψη {path.name} ({len(s)} βαθμολογίες αντί για {n})")
        return None
    return s


def fmt(metric, v):
    """Μορφοποίηση τιμής: ποσοστό με ένα δεκαδικό ή αριθμός με τρία."""
    if v is None or np.isnan(v):
        return "—"
    return f"{100 * v:.1f}%" if metric in PERCENT else f"{v:.3f}"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def print_table(title, header, rows):
    print(f"\n{title}")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for row in [header] + rows:
        print("  " + "  ".join(str(x).ljust(w) for x, w in zip(row, widths)))


# ===========================================================================
# 2. Υπολογισμός μετρικών για ένα σύνολο βαθμολογιών
# ===========================================================================
def compute(y, types, s, rem, probabilistic):
    """
    y, types, s : ετικέτες, τύποι και βαθμολογίες των ζευγών του test
    rem         : βαθμολογίες των ζευγών «νέα μελωδία, ίδιος ρυθμός» (ή None)
    probabilistic: True αν οι βαθμολογίες είναι στο [0, 1] (Siamese), ώστε να έχει νόημα το threshold 0.5
    """
    out = {m: np.nan for m in METRICS}
    out["auc"] = roc_auc_score(y, s)
    out["ap"] = average_precision_score(y, s)
    out["tpr1"] = tpr_at_fpr(y, s)
    mask = (types == "partial") | (y == 0)
    out["auc_partial"] = roc_auc_score(y[mask], s[mask])
    if rem is not None and len(rem):
        pos, neg = s[y == 1], s[y == 0]
        out["auc_rem"] = roc_auc_score(
            np.concatenate([np.ones(len(pos)), np.zeros(len(rem))]),
            np.concatenate([pos, rem]))
        thr = np.quantile(neg, 0.99)   # decision threshold με 1% ψευδείς ενδείξεις
        out["fp_rem_1"] = float((rem > thr).mean())
        if probabilistic:
            out["fp_rem_05"] = float((rem > 0.5).mean())
    return out


# ===========================================================================
# 3. (Α) Επαναλήψεις με διαφορετικό seed
# ===========================================================================
def find_runs(base):
    """Ονόματα εκτελέσεων ενός μοντέλου: η αρχική και όσες έχουν επίθημα _s<αριθμός>."""
    runs = []
    if (config.RESULTS_DIR / f"scores_{base}_test.npy").exists():
        runs.append(base)
    pattern = re.compile(rf"^scores_{re.escape(base)}_s(\d+)_test\.npy$")
    extra = sorted((int(m.group(1)), p.name) for p in config.RESULTS_DIR.glob("scores_*_test.npy")
                   if (m := pattern.match(p.name)))
    runs += [f"{base}_s{seed}" for seed, _ in extra]
    return runs


def branches_of(name, k):
    """Οι κλάδοι του μοντέλου (από το .json της εκπαίδευσης)."""
    path = config.MODELS_DIR / f"{name}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)["branches"]
    return ["melody", "rhythm", "harmony"][:k]


def seeds_analysis(test, rem_n):
    y, types = test["label"], test["type"]
    summary, per_run, contrib_rows = [], [], []
    for base in SIAMESE:
        runs = find_runs(base)
        if not runs:
            continue
        values = {m: [] for m in METRICS}
        contribs = {}
        for name in runs:
            s = load(name, "test", len(y))
            if s is None:
                continue
            rem = load(name, "test_remelodized", rem_n) if rem_n else None
            r = compute(y, types, s, rem, probabilistic=True)
            for m in METRICS:
                values[m].append(r[m])
            per_run.append([name] + [fmt(m, r[m]) for m in METRICS])

            # Μέση συνεισφορά κάθε κλάδου (logit) σε θετικά/αρνητικά ζεύγη.
            cpath = config.RESULTS_DIR / f"contrib_{name}_test.npy"
            if cpath.exists():
                c = np.load(cpath)
                if len(c) == len(y):
                    for b, col in zip(branches_of(name, c.shape[1]), c.T):
                        contribs.setdefault((b, 1), []).append(col[y == 1].mean())
                        contribs.setdefault((b, 0), []).append(col[y == 0].mean())

        n = len(values["auc"])
        if n == 0:
            continue
        row = [S.METHOD_LABELS[base], str(n)]
        for m in METRICS:
            v = np.array(values[m], dtype=float)
            if np.isnan(v).all():
                row.append("—")
            elif n == 1:
                row.append(fmt(m, v[0]))
            else:
                sd = v.std(ddof=1)
                row.append(f"{100 * v.mean():.1f} ± {100 * sd:.1f}%" if m in PERCENT
                           else f"{v.mean():.3f} ± {sd:.3f}")
        summary.append(row)

        order = ["melody", "rhythm", "harmony"]
        for (b, label), v in sorted(contribs.items(),
                                    key=lambda kv: (-kv[0][1], order.index(kv[0][0]))):
            v = np.array(v)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            contrib_rows.append([S.METHOD_LABELS[base], "θετικά" if label else "αρνητικά",
                                 b, f"{v.mean():+.2f} ± {sd:.2f}"])

    header = ["Μοντέλο", "Εκτελέσεις"] + [METRIC_LABELS[m] for m in METRICS]
    print_table("Πίνακας 5 — Μέσος όρος ± τυπική απόκλιση σε επαναλήψεις με "
                "διαφορετικό seed", header, summary)
    write_csv(config.RESULTS_DIR / "table_seeds.csv", header, summary)

    header_r = ["Εκτέλεση"] + [METRIC_LABELS[m] for m in METRICS]
    print_table("Ανά εκτέλεση", header_r, per_run)
    write_csv(config.RESULTS_DIR / "table_seeds_runs.csv", header_r, per_run)

    if contrib_rows:
        header_c = ["Μοντέλο", "Ζεύγη", "Κλάδος", "Μέση συνεισφορά ± τ.α."]
        print_table("Μέση συνεισφορά κλάδων (logit) σε όλες τις εκτελέσεις",
                    header_c, contrib_rows)
        write_csv(config.RESULTS_DIR / "table_seeds_contrib.csv", header_c, contrib_rows)

    if summary and all(r[1] == "1" for r in summary):
        print("\nΒρέθηκε μόνο μία εκτέλεση ανά μοντέλο. Τρέξτε πρώτα τις επαναλήψεις "
              "του train.py με --seed (βλ. README).")


# ===========================================================================
# 4. (Β) Bootstrap σε επίπεδο τραγουδιού
# ===========================================================================
def bootstrap_analysis(test, rem_data, n_boot):
    y, types = test["label"], test["type"]
    songs = np.unique(test["song_a"])
    idx_test = {g: np.flatnonzero(test["song_a"] == g) for g in songs}
    idx_rem = ({g: np.flatnonzero(rem_data["song_a"] == g) for g in songs}
               if rem_data is not None else None)
    rng = np.random.default_rng(config.SEED)
    # Ίδια δείγματα τραγουδιών για όλες τις μεθόδους (ζευγαρωτή σύγκριση).
    samples = [rng.choice(songs, size=len(songs), replace=True) for _ in range(n_boot)]

    rows = []
    for name in S.METHOD_ORDER:
        s = load(name, "test", len(y))
        if s is None:
            continue
        rem = load(name, "test_remelodized", len(rem_data["label"])) if rem_data is not None else None
        prob = name.startswith("siamese")
        point = compute(y, types, s, rem, prob)
        boots = {m: [] for m in METRICS}
        for sample in samples:
            i = np.concatenate([idx_test[g] for g in sample])
            if len(np.unique(y[i])) < 2 or not ((types[i] == "partial").any()):
                continue
            j = (np.concatenate([idx_rem[g] for g in sample])
                 if rem is not None else None)
            r = compute(y[i], types[i], s[i], rem[j] if rem is not None else None, prob)
            for m in METRICS:
                boots[m].append(r[m])
        row = [S.METHOD_LABELS[name]]
        for m in METRICS:
            b = np.array(boots[m], dtype=float)
            if np.isnan(point[m]):
                row.append("—")
                continue
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            row.append(f"{fmt(m, point[m])} [{fmt(m, lo)}, {fmt(m, hi)}]")
        rows.append(row)
        print(f"  bootstrap: {S.METHOD_LABELS[name]} ολοκληρώθηκε")

    header = ["Μέθοδος"] + [METRIC_LABELS[m] for m in METRICS]
    print_table(f"Πίνακας 6 — Τιμή και διάστημα εμπιστοσύνης 95% (bootstrap "
                f"{n_boot} επαναλήψεων, {len(songs)} τραγούδια)", header, rows)
    write_csv(config.RESULTS_DIR / "table_bootstrap.csv", header, rows)


def main():
    parser = argparse.ArgumentParser(description="Επαναλήψεις με διαφορετικό seed και bootstrap")
    parser.add_argument("--n_boot", type=int, default=1000,
                        help="πλήθος επαναλήψεων bootstrap (προεπιλογή 1000)")
    parser.add_argument("--skip_bootstrap", action="store_true")
    args = parser.parse_args()
    config.ensure_dirs()

    test = PR.load_pairs("test")
    rem_path = config.PAIRS_DIR / "test_remelodized.npz"
    rem_data = PR.load_pairs("test_remelodized") if rem_path.exists() else None
    rem_n = len(rem_data["label"]) if rem_data is not None else 0

    seeds_analysis(test, rem_n)
    if not args.skip_bootstrap:
        print(f"\nBootstrap ({args.n_boot} επαναλήψεις) — μπορεί να διαρκέσει μερικά λεπτά...")
        bootstrap_analysis(test, rem_data, args.n_boot)
    print(f"\nΠίνακες: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
