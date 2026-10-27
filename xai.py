# -*- coding: utf-8 -*-
"""
xai.py
======
Επεξηγησιμότητα (Explainable AI) του Siamese ανιχνευτή.

Το μοντέλο εξηγεί κάθε απόφαση σε τρία επίπεδα:

1. ΣΕ ΤΙ μοιάζουν τα δύο τμήματα: η βαθμολογία ομοιότητας είναι
   p = σ(c_μελωδία + c_ρυθμός + c_αρμονία), οπότε οι συνεισφορές c_k
   διαβάζονται απευθείας (ερμηνευσιμότητα «εκ κατασκευής»).
2. ΠΟΥ μοιάζουν: οι πίνακες ομοιότητας (32 × 32 χρόνοι) δείχνουν ποιο σημείο
   του Α αντιστοιχεί σε ποιο σημείο του Β· μια διαγώνια γραμμή δείχνει
   αντιγραμμένο απόσπασμα.
3. ΠΟΙΕΣ ΝΟΤΕΣ οδήγησαν στην απόφαση: με τη μέθοδο Integrated Gradients
   (Sundararajan, Taly & Yan, 2017) η απόφαση κατανέμεται σε κάθε χρονικό
   βήμα της εισόδου. Η μέθοδος ικανοποιεί την ιδιότητα της πληρότητας
   (completeness): το άθροισμα των αποδόσεων ισούται με τη διαφορά της
   εξόδου από την έξοδο της αναφοράς (baseline). Την ελέγχουμε αριθμητικά.

Εκτέλεση:
    python xai.py                          (μοντέλο v2, ένα παράδειγμα ανά τύπο ζεύγους)
    python xai.py --name siamese           (για άλλο εκπαιδευμένο μοντέλο, π.χ. v1)
    python xai.py --index 17 42            (συγκεκριμένα ζεύγη του test set)
"""

import argparse
import csv
import json

import numpy as np
import tensorflow as tf
from matplotlib.colors import TwoSlopeNorm

import config
import model as M
import pairs as PR
import plot_style as S

plt = S.plt
T = config.SEGMENT_STEPS
SPB = config.STEPS_PER_BEAT


# ===========================================================================
# 1. Φόρτωση εκπαιδευμένου μοντέλου
# ===========================================================================
def load_trained(name):
    """Ξαναχτίζει το μοντέλο και φορτώνει τα εκπαιδευμένα βάρη."""
    with open(config.MODELS_DIR / f"{name}.json", encoding="utf-8") as f:
        branches = tuple(json.load(f)["branches"])
    detector, explainer = M.build_models(branches)
    detector.load_weights(str(config.MODELS_DIR / f"{name}.weights.h5"))
    return detector, explainer, branches


def pair_inputs(data, idx, branches):
    """Εισοδοι του μοντέλου για ένα ζεύγος (παρτίδα μεγέθους 1)."""
    one = {k: v[idx:idx + 1] for k, v in data.items() if k.startswith(("a_", "b_"))}
    return M.select_inputs(M.to_model_inputs(one), branches)


# ===========================================================================
# 2. Integrated Gradients
# ===========================================================================
def integrated_gradients(explainer, inputs, steps=64):
    """
    Integrated Gradients ως προς το logit (άθροισμα συνεισφορών).

    Αναφορά (baseline): μηδενική είσοδος, δηλαδή «καμία μουσική πληροφορία».
    Το ολοκλήρωμα των κλίσεων κατά μήκος της ευθείας baseline → είσοδος
    προσεγγίζεται με τον κανόνα του τραπεζίου σε `steps` διαστήματα.

    Επιστρέφει:
        attributions : λεξικό όνομα_εισόδου → (T, κανάλια) αποδόσεις
        delta        : logit(είσοδος) − logit(baseline)
        total        : άθροισμα όλων των αποδόσεων (≈ delta, έλεγχος πληρότητας)
    """
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1), (-1, 1, 1))
    scaled = {k: alphas * tf.cast(v, tf.float32) for k, v in inputs.items()}
    with tf.GradientTape() as tape:
        tape.watch(list(scaled.values()))
        logit = tf.reduce_sum(explainer(scaled, training=False)["contrib"], axis=-1)
    grads = tape.gradient(logit, scaled)

    attributions, total = {}, 0.0
    for k, x in inputs.items():
        g = grads[k]
        avg_grad = tf.reduce_mean((g[:-1] + g[1:]) / 2.0, axis=0)   # τραπέζιο
        attr = (tf.cast(x[0], tf.float32) * avg_grad).numpy()
        attributions[k] = attr
        total += attr.sum()
    delta = float(logit[-1] - logit[0])
    return attributions, delta, float(total)


def time_attribution(attributions, side, branches):
    """Άθροισμα αποδόσεων όλων των κλάδων ανά χρονικό βήμα για μία πλευρά (a/b)."""
    return sum(attributions[f"{side}_{b}"].sum(axis=1) for b in branches)


# ===========================================================================
# 3. Οπτικοποίηση
# ===========================================================================
def draw_piano_roll(ax, notes, step_attr, norm, title):
    """
    Piano-roll της μελωδίας: κάθε νότα είναι οριζόντια μπάρα (χρόνος × ύψος),
    χρωματισμένη με τη μέση απόδοση IG στη διάρκειά της (κόκκινο = οδήγησε
    προς «αντιγραφή», μπλε = προς «όχι αντιγραφή»).
    """
    cmap = plt.get_cmap("RdBu_r")
    for on, dur, pitch in notes:
        end = min(T, on + dur)
        value = step_attr[on:end].mean() if end > on else 0.0
        ax.barh(pitch, (end - on) / SPB, left=on / SPB, height=0.8,
                color=cmap(norm(value)), edgecolor=S.MUTED, linewidth=0.4)
    ax.set_xlim(0, config.SEGMENT_BEATS)
    ax.set_xticks(range(0, config.SEGMENT_BEATS + 1, 4))
    ax.set_xlabel("Χρόνος (beats)")
    ax.set_ylabel("Τονικό ύψος (MIDI)")
    ax.set_title(title)


def plot_example(pair, out, attributions, branches, info, path):
    """Δημιουργεί την εικόνα επεξήγησης για ένα ζεύγος."""
    attr_a = time_attribution(attributions, "a", branches)
    attr_b = time_attribution(attributions, "b", branches)
    vmax = max(np.abs(attr_a).max(), np.abs(attr_b).max(), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    n = len(branches)
    fig = plt.figure(figsize=(3.8 * max(n, 2), 8.0))
    grid = fig.add_gridspec(2, max(n, 2) * 2, height_ratios=[1, 1.05],
                            hspace=0.5, wspace=1.2)
    half = max(n, 2)
    ax_a = fig.add_subplot(grid[0, :half])
    ax_b = fig.add_subplot(grid[0, half:])
    draw_piano_roll(ax_a, pair["a"]["melody"], attr_a, norm, "Τμήμα Α (αναφορά)")
    draw_piano_roll(ax_b, pair["b"]["melody"], attr_b, norm, "Τμήμα Β (υπό εξέταση)")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    fig.colorbar(sm, ax=[ax_a, ax_b], fraction=0.02, pad=0.01,
                 label="Απόδοση IG (→ αντιγραφή)")

    contrib = np.atleast_1d(np.asarray(out["contrib"])[0])
    width = (max(n, 2) * 2) // n
    for i, b in enumerate(branches):
        ax = fig.add_subplot(grid[1, i * width:(i + 1) * width])
        sim = np.asarray(out[f"sim_{b}"])[0]
        im = ax.imshow(sim, origin="lower", cmap="Blues", vmin=0, vmax=1,
                       extent=[0, config.SEGMENT_BEATS, 0, config.SEGMENT_BEATS])
        ax.set_title(f"{S.BRANCH_LABELS[b]}: συνεισφορά {contrib[i]:+.1f}")
        ax.set_xlabel("Β: χρόνος (beats)")
        ax.set_ylabel("Α: χρόνος (beats)")
        ax.grid(False)
    fig.colorbar(im, ax=fig.axes[-n:], fraction=0.02, pad=0.01,
                 label="Ομοιότητα συνημιτόνου")
    fig.suptitle(info, fontsize=11)
    fig.savefig(path)
    plt.close(fig)


def plot_contributions(contrib, types, branches, path_png, path_csv):
    """
    Μέση συνεισφορά κάθε διάστασης ανά τύπο ζεύγους (test set).
    Δείχνει σε ποια μουσική διάσταση στηρίζεται το μοντέλο για κάθε τύπο.
    """
    kinds = PR.POSITIVE_TYPES + PR.NEGATIVE_TYPES
    means = np.array([contrib[types == k].mean(axis=0) for k in kinds])
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "n"] + list(branches))
        for k, row in zip(kinds, means):
            w.writerow([k, int((types == k).sum())] + [f"{v:.3f}" for v in row])

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(kinds))
    width = 0.8 / len(branches)
    for i, b in enumerate(branches):
        ax.bar(x + (i - (len(branches) - 1) / 2) * width, means[:, i], width * 0.92,
               color=S.BRANCH_COLORS[b], label=S.BRANCH_LABELS[b])
    ax.axhline(0, color=S.MUTED, linewidth=0.8)
    ax.axvline(len(PR.POSITIVE_TYPES) - 0.5, color=S.MUTED, linewidth=0.8, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([S.TYPE_LABELS.get(k, k) for k in kinds], rotation=30, ha="right")
    ax.set_axisbelow(True)
    ax.set_ylabel("Μέση συνεισφορά (logit)")
    ax.set_title("Συνεισφορά κάθε μουσικής διάστασης ανά τύπο ζεύγους", pad=24)
    ymax = ax.get_ylim()[1]
    ax.text(len(PR.POSITIVE_TYPES) / 2 - 0.5, ymax, "ζεύγη αντιγραφής", ha="center",
            va="bottom", color=S.MUTED, fontsize=9)
    ax.text(len(PR.POSITIVE_TYPES) + len(PR.NEGATIVE_TYPES) / 2 - 0.5, ymax,
            "ανεξάρτητα έργα", ha="center", va="bottom", color=S.MUTED, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(path_png)
    plt.close(fig)


# ===========================================================================
# 4. Κύρια διαδικασία
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Επεξηγήσεις του ανιχνευτή")
    parser.add_argument("--name", default="siamese_v2", help="όνομα εκπαιδευμένου μοντέλου")
    parser.add_argument("--index", type=int, nargs="*",
                        help="δείκτες ζευγών του test set (προεπιλογή: ένα ανά τύπο)")
    args = parser.parse_args()
    config.ensure_dirs()

    _, explainer, branches = load_trained(args.name)
    data, notes = PR.load_pairs("test"), PR.load_pair_notes("test")

    if args.index:
        indices = args.index
    else:   # το πρώτο ζεύγος κάθε τύπου
        kinds = PR.POSITIVE_TYPES + PR.NEGATIVE_TYPES
        indices = [int(np.flatnonzero(data["type"] == k)[0]) for k in kinds
                   if (data["type"] == k).any()]

    print("Έλεγχος πληρότητας IG: άθροισμα αποδόσεων ≈ logit(x) − logit(baseline)")
    for idx in indices:
        inputs = pair_inputs(data, idx, branches)
        out = explainer(inputs, training=False)
        attributions, delta, total = integrated_gradients(explainer, inputs)
        prob = float(np.asarray(out["prob"]).ravel()[0])
        kind, label = data["type"][idx], int(data["label"][idx])
        info = (f"Ζεύγος #{idx} — τύπος: {S.TYPE_LABELS.get(kind, kind)} "
                f"({'αντιγραφή' if label == 1 else 'ανεξάρτητα'}) — "
                f"βαθμολογία ομοιότητας: {prob:.3f}")
        path = config.FIGURES_DIR / f"xai_{args.name}_{kind}_{idx}.png"
        plot_example(notes[idx], out, attributions, branches, info, path)
        print(f"  #{idx:5d} {kind:12s} p={prob:.3f}  Δlogit={delta:+8.3f}  "
              f"ΣIG={total:+8.3f}  → {path.name}")

    contrib = np.load(config.RESULTS_DIR / f"contrib_{args.name}_test.npy")
    plot_contributions(contrib.reshape(len(contrib), -1), data["type"], branches,
                       config.FIGURES_DIR / f"contributions_{args.name}.png",
                       config.RESULTS_DIR / f"contributions_{args.name}.csv")
    print(f"Οι εικόνες αποθηκεύτηκαν στο {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
