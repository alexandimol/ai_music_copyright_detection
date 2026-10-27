# -*- coding: utf-8 -*-
"""
memorization.py
===============
Πείραμα: αντιγράφει ο generator μελωδίες από τα δεδομένα εκπαίδευσής του;

Για κάθε παραγόμενη μελωδία αναζητείται το πιο όμοιο τμήμα του συνόλου
εκπαίδευσης του generator, σε δύο στάδια:
    1. Προεπιλογή υποψηφίων: τα MEM_CANDIDATES τμήματα με τα περισσότερα κοινά
       n-grams μελωδικών διαστημάτων (γρήγορη αναζήτηση με ευρετήριο).
    2. Πλήρης σύγκριση των υποψηφίων με (α) τον εκπαιδευμένο ανιχνευτή,
       (β) τον αλγόριθμο Smith–Waterman και (γ) το μήκος του μεγαλύτερου
       ΑΥΤΟΥΣΙΟΥ κοινού αποσπάσματος (σε νότες).

Ομάδες που συγκρίνονται:
    gen_full   : μελωδίες του generator που εκπαιδεύτηκε σε όλα τα δεδομένα
    gen_small  : μελωδίες του generator με υπερπροσαρμογή σε 100 τμήματα
    human_test : ομάδα ελέγχου (control group) — ανθρώπινες μελωδίες από τραγούδια του test set,
                 που ΔΕΝ ανήκουν στα δεδομένα εκπαίδευσης. Δείχνει πόση
                 «ομοιότητα» εμφανίζεται φυσιολογικά μεταξύ ανεξάρτητων έργων
                 (κοινά μουσικά στοιχεία), δηλαδή το ποσοστό ψευδών ενδείξεων.

Χρησιμοποιείται ο ανιχνευτής «μόνο μελωδίας» (siamese_melody_v2, ή
siamese_melody αν δεν υπάρχει η v2), επειδή ο generator παράγει μελωδίες
χωρίς συνοδεία.

Εκτέλεση (αφού τρέξουν τα generator.py και train.py --branches melody):
    python memorization.py
"""

import argparse
import csv
import json
import pickle
import random
from collections import Counter, defaultdict

import numpy as np

import config
import pairs as PR
import preprocessing as P
import plot_style as S
import xai
from baselines import note_intervals, smith_waterman
from train import make_dataset

plt = S.plt
GROUP_LABELS = {"gen_full": "Generator (όλα τα δεδομένα)",
                "gen_small": "Generator (λίγα δεδομένα, υπερπροσαρμογή)",
                "human_test": "Ανθρώπινες μελωδίες (έλεγχος)"}


# ===========================================================================
# 1. Αναζήτηση υποψηφίων με n-grams διαστημάτων
# ===========================================================================
def interval_ngrams(seg, n=config.MEM_NGRAM):
    """Σύνολο n-grams μελωδικών διαστημάτων (ανεξάρτητα από τονικότητα)."""
    iv = note_intervals(seg["melody"])
    return {tuple(iv[i:i + n]) for i in range(len(iv) - n + 1)}


def build_index(refs):
    """Ανεστραμμένο ευρετήριο: n-gram → λίστα τμημάτων αναφοράς που το περιέχουν."""
    index = defaultdict(list)
    for i, seg in enumerate(refs):
        for g in interval_ngrams(seg):
            index[g].append(i)
    return index


def candidates(query, index, n_refs, k, rng):
    """Τα k τμήματα αναφοράς με τα περισσότερα κοινά n-grams με το query."""
    counts = Counter()
    for g in interval_ngrams(query):
        counts.update(index.get(g, ()))
    top = [i for i, _ in counts.most_common(k)]
    while len(top) < k:                       # συμπλήρωση αν βρέθηκαν λίγα
        top.append(rng.randrange(n_refs))
    return top


def longest_common_run(x, y):
    """Μήκος του μεγαλύτερου κοινού συνεχούς τμήματος δύο ακολουθιών διαστημάτων."""
    best, prev = 0, [0] * (len(y) + 1)
    for xi in x:
        cur = [0] * (len(y) + 1)
        for j, yj in enumerate(y, 1):
            if xi == yj:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


# ===========================================================================
# 2. Σύγκριση μιας ομάδας με τα δεδομένα αναφοράς
# ===========================================================================
def analyse_group(queries, refs, detector, branches, rng):
    """
    Για κάθε μελωδία της ομάδας βρίσκει το πιο όμοιο τμήμα αναφοράς.
    Επιστρέφει λίστα αποτελεσμάτων (ένα λεξικό ανά μελωδία).
    """
    index = build_index(refs)
    pairs_list, owner = [], []
    for qi, q in enumerate(queries):
        for ri in candidates(q, index, len(refs), config.MEM_CANDIDATES, rng):
            pairs_list.append({"a": refs[ri], "b": q, "label": 0,
                               "type": "memorization", "degree": 0.0, "ref": ri})
            owner.append(qi)

    data = PR.encode_pairs(pairs_list)
    probs = detector.predict(make_dataset(data, branches), verbose=0).ravel()

    results = []
    owner = np.array(owner)
    for qi, q in enumerate(queries):
        idx = np.flatnonzero(owner == qi)
        best = idx[np.argmax(probs[idx])]
        ref = refs[pairs_list[best]["ref"]]
        iq, ir = note_intervals(q["melody"]), note_intervals(ref["melody"])
        sw = max(smith_waterman(iq, note_intervals(refs[pairs_list[j]["ref"]]["melody"]))
                 for j in idx)
        results.append({
            "query": q["song_id"], "match_song": ref["song_id"],
            "match_seg": ref["seg_idx"], "prob": float(probs[best]),
            "smith_waterman": float(sw),
            "exact_run_notes": longest_common_run(iq, ir) + 1,  # διαστήματα → νότες
            "pair": pairs_list[best],
        })
    return results


def summarize(name, results):
    """Συγκεντρωτικά στατιστικά μιας ομάδας."""
    prob = np.array([r["prob"] for r in results])
    run = np.array([r["exact_run_notes"] for r in results])
    return {
        "group": name, "n": len(results),
        "mean_max_prob": float(prob.mean()),
        "flagged_%": float(100 * (prob > config.MEM_THRESHOLD).mean()),
        "median_smith_waterman": float(np.median([r["smith_waterman"] for r in results])),
        "exact_run_>=8_notes_%": float(100 * (run >= 8).mean()),
        "exact_run_>=16_notes_%": float(100 * (run >= 16).mean()),
        "median_exact_run_notes": float(np.median(run)),
    }


# ===========================================================================
# 3. Γραφήματα
# ===========================================================================
def plot_distributions(all_results, path):
    """
    Αθροιστική κατανομή (ECDF) της μέγιστης βαθμολογίας ομοιότητας ανά ομάδα,
    και του μήκους του μεγαλύτερου αυτούσιου κοινού αποσπάσματος.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for i, (name, results) in enumerate(all_results.items()):
        color, marker = S.COLORS[i], S.MARKERS[i]
        for ax, key in zip(axes, ("prob", "exact_run_notes")):
            v = np.sort([r[key] for r in results])
            y = np.arange(1, len(v) + 1) / len(v)
            ax.step(v, y, where="post", color=color, label=GROUP_LABELS.get(name, name),
                    linestyle=S.LINESTYLES[i])
            ax.plot(v[::max(1, len(v) // 8)], y[::max(1, len(v) // 8)], marker,
                    color=color, linestyle="none")
    axes[0].axvline(config.MEM_THRESHOLD, color=S.MUTED, linestyle=":", linewidth=1)
    axes[0].set_xlabel("Μέγιστη βαθμολογία ομοιότητας (ανιχνευτής)")
    axes[1].set_xlabel("Μεγαλύτερο αυτούσιο κοινό απόσπασμα (νότες)")
    for ax in axes:
        ax.set_ylabel("Αθροιστικό ποσοστό μελωδιών")
        ax.set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ομοιότητα με το πλησιέστερο τμήμα των δεδομένων εκπαίδευσης")
    fig.savefig(path)
    plt.close(fig)


# ===========================================================================
# 4. Κύρια διαδικασία
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Πείραμα απομνημόνευσης")
    parser.add_argument("--model", default=None,
                        help="ανιχνευτής (προεπιλογή: siamese_melody_v2 αν υπάρχει)")
    parser.add_argument("--generators", nargs="*", default=["gen_full", "gen_small"])
    parser.add_argument("--n_control", type=int, default=config.GEN_N_SAMPLES)
    args = parser.parse_args()
    config.ensure_dirs()
    rng = random.Random(config.SEED)
    if args.model is None:
        v2 = config.MODELS_DIR / "siamese_melody_v2.weights.h5"
        args.model = "siamese_melody_v2" if v2.exists() else "siamese_melody"
    print(f"Ανιχνευτής: {args.model}")

    detector, explainer, branches = xai.load_trained(args.model)
    segments, splits = P.load_segments()
    by_key = {(s["song_id"], s["seg_idx"]): s for s in segments}
    train_ids = set(splits["train"])
    all_train = [s for s in segments if s["song_id"] in train_ids]

    groups = {}
    for name in args.generators:
        path = config.GENERATED_DIR / f"{name}.pkl"
        if not path.exists():
            print(f"Παράλειψη '{name}': δεν βρέθηκε το {path} (τρέξτε το generator.py)")
            continue
        with open(path, "rb") as f:
            queries = pickle.load(f)
        with open(config.GENERATED_DIR / f"{name}_train_ids.json", encoding="utf-8") as f:
            refs = [by_key[tuple(k)] for k in json.load(f)["train"]]
        groups[name] = (queries, refs)
    test_ids = set(splits["test"])
    human = [s for s in segments if s["song_id"] in test_ids]
    groups["human_test"] = (rng.sample(human, min(args.n_control, len(human))), all_train)

    all_results, summaries = {}, []
    for name, (queries, refs) in groups.items():
        print(f"\n{name}: {len(queries)} μελωδίες έναντι {len(refs)} τμημάτων αναφοράς")
        results = analyse_group(queries, refs, detector, branches, rng)
        all_results[name] = results
        summaries.append(summarize(name, results))
        for k, v in summaries[-1].items():
            print(f"   {k:24s} {v:.2f}" if isinstance(v, float) else f"   {k:24s} {v}")
        with open(config.RESULTS_DIR / f"memorization_{name}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query", "match_song", "match_seg", "prob", "smith_waterman",
                        "exact_run_notes"])
            for r in sorted(results, key=lambda r: -r["prob"]):
                w.writerow([r["query"], r["match_song"], r["match_seg"],
                            f"{r['prob']:.4f}", f"{r['smith_waterman']:.4f}",
                            r["exact_run_notes"]])

    with open(config.RESULTS_DIR / "memorization_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    plot_distributions(all_results, config.FIGURES_DIR / "memorization_ecdf.png")

    # Επεξήγηση (XAI) για τις 3 πιο «ύποπτες» μελωδίες κάθε generator.
    for name, results in all_results.items():
        if name == "human_test":
            continue
        for rank, r in enumerate(sorted(results, key=lambda r: -r["prob"])[:3], 1):
            data = PR.encode_pairs([r["pair"]])
            inputs = xai.pair_inputs(data, 0, branches)
            out = explainer(inputs, training=False)
            attributions, _, _ = xai.integrated_gradients(explainer, inputs)
            info = (f"{GROUP_LABELS.get(name, name)} — μελωδία {r['query']} έναντι "
                    f"τραγουδιού {r['match_song']} — βαθμολογία ομοιότητας {r['prob']:.3f}")
            xai.plot_example(r["pair"], out, attributions, branches, info,
                             config.FIGURES_DIR / f"memorization_{name}_top{rank}.png")
    print(f"\nΑποτελέσματα: {config.RESULTS_DIR}  |  Εικόνες: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
