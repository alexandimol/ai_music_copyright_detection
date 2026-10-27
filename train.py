# -*- coding: utf-8 -*-
"""
train.py
========
Εκπαίδευση και αξιολόγηση του Siamese ανιχνευτή (model.py).

Βήματα:
    1. Φόρτωση των ζευγών train/val από το pairs.py.
    2. Εκπαίδευση με δυαδική διασταυρούμενη εντροπία (binary cross-entropy),
       βελτιστοποιητή Adam και πρόωρο τερματισμό (early stopping) με βάση
       το ROC-AUC στο σύνολο επικύρωσης.
    3. Πρόβλεψη στα test και sweep σύνολα και αποθήκευση των βαθμολογιών στο
       outputs/results/, στην ίδια μορφή με τα baselines, ώστε το evaluate.py
       να τα συγκρίνει.

Εκτέλεση:
    python train.py --quick                    (γρήγορη δοκιμή, 1–3 λεπτά)
    python train.py                            (πλήρης εκπαίδευση, όλοι οι κλάδοι)
    python train.py --branches melody --name siamese_melody
                                               (πείραμα αφαίρεσης: μόνο μελωδία)
    python train.py --name siamese_v2 --extra remelodized
                                               (έκδοση v2 με τα επιπλέον αρνητικά)
    python train.py --name siamese --eval_only (μόνο αξιολόγηση ήδη εκπαιδευμένου)
    python train.py --name siamese_v2 --extra remelodized --seed 1
                                               (επανάληψη με άλλο seed· αποθηκεύεται
                                                ως siamese_v2_s1, για το robustness.py)

Σημείωση για το seed (--seed): αλλάζει μόνο την αρχικοποίηση των βαρών, τη
σειρά των παραδειγμάτων και την τυχαία ανταλλαγή Α/Β κατά την εκπαίδευση.
Τα ζεύγη (δεδομένα) μένουν ίδια, ώστε οι διαφορές ανάμεσα στις επαναλήψεις να
οφείλονται μόνο στη στοχαστικότητα της εκπαίδευσης.
"""

import argparse
import json
import time

import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import config
import model as M
import pairs as PR

EVAL_SPLITS = ["test", "sweep_melodic", "sweep_partial", "test_remelodized"]


# ===========================================================================
# 1. Ροή δεδομένων (tf.data)
# ===========================================================================
def make_dataset(data, branches, shuffle=False, augment=False,
                 batch_size=config.BATCH_SIZE, seed=config.SEED):
    """
    Δημιουργεί tf.data.Dataset από τα αποθηκευμένα ζεύγη. Η μετατροπή σε
    one-hot γίνεται «στη ροή» (ανά παρτίδα), ώστε να μη χρειάζεται να
    αποθηκευτούν στη μνήμη όλα τα ζεύγη σε one-hot μορφή.
    """
    fields = {k: v for k, v in data.items() if k.startswith(("a_", "b_"))}
    labels = data["label"].astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((fields, labels))
    if shuffle:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    def prepare(x, y):
        inputs = M.to_model_inputs(x)
        if augment:
            # Με πιθανότητα 50% ανταλλάσσουμε τα Α και Β της παρτίδας.
            inputs = tf.cond(tf.random.uniform([]) < 0.5,
                             lambda: M.swap_sides(inputs), lambda: inputs)
        return M.select_inputs(inputs, branches), y

    return ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def concat(a, b):
    """Ενώνει δύο σύνολα ζευγών (ίδια πεδία)."""
    return {k: np.concatenate([a[k], b[k]]) for k in a}


def take(data, n):
    """Κρατά τα πρώτα n ζεύγη (για γρήγορη δοκιμή)."""
    return {k: v[:n] for k, v in data.items()}


# ===========================================================================
# 2. Εκπαίδευση
# ===========================================================================
def train(name, branches, epochs, quick, extra=(), seed=config.SEED):
    # Το seed ορίζει την αρχικοποίηση των βαρών και κάθε τυχαία επιλογή
    # του TensorFlow/NumPy/Python κατά την εκπαίδευση.
    keras.utils.set_random_seed(seed)
    train_data, val_data = PR.load_pairs("train"), PR.load_pairs("val")
    for kind in extra:   # επιπλέον αρνητικά (έκδοση v2)
        train_data = concat(train_data, PR.load_pairs(f"train_{kind}"))
        val_data = concat(val_data, PR.load_pairs(f"val_{kind}"))
    if quick:
        train_data, val_data = take(train_data, 2000), take(val_data, 500)
    print(f"Εκπαίδευση '{name}' (seed {seed}) με κλάδους {branches}: "
          f"{len(train_data['label'])} ζεύγη εκπαίδευσης, "
          f"{len(val_data['label'])} επικύρωσης")

    detector, explainer = M.build_models(branches)
    detector.compile(
        optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"),
                 keras.metrics.BinaryAccuracy(name="accuracy")])
    detector.summary()

    weights_path = config.MODELS_DIR / f"{name}.weights.h5"
    callbacks = [
        # Σταματά όταν το val AUC δεν βελτιώνεται για 5 εποχές και επαναφέρει
        # τα καλύτερα βάρη.
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                      restore_best_weights=True),
        # Μειώνει τον ρυθμό μάθησης στο μισό όταν η βελτίωση «κολλήσει».
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                          factor=0.5, patience=2),
        keras.callbacks.ModelCheckpoint(str(weights_path), monitor="val_auc",
                                        mode="max", save_best_only=True,
                                        save_weights_only=True),
        keras.callbacks.CSVLogger(str(config.RESULTS_DIR / f"history_{name}.csv")),
    ]
    t0 = time.time()
    detector.fit(make_dataset(train_data, branches, shuffle=True, augment=True, seed=seed),
                 validation_data=make_dataset(val_data, branches),
                 epochs=epochs, callbacks=callbacks, verbose=2)
    print(f"Χρόνος εκπαίδευσης: {(time.time() - t0) / 60:.1f} λεπτά")
    detector.save_weights(str(weights_path))   # τα καλύτερα βάρη (restore_best_weights)

    with open(config.MODELS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"branches": list(branches), "extra": list(extra), "seed": seed}, f)
    return explainer


# ===========================================================================
# 3. Αξιολόγηση και αποθήκευση βαθμολογιών
# ===========================================================================
def evaluate(explainer, name, branches, quick):
    for split in EVAL_SPLITS:
        if not (config.PAIRS_DIR / f"{split}.npz").exists():
            continue
        data = PR.load_pairs(split)
        if quick:
            data = take(data, 200)
        out = explainer.predict(make_dataset(data, branches), verbose=0)
        prob = np.asarray(out["prob"]).ravel()
        contrib = np.asarray(out["contrib"])
        np.save(config.RESULTS_DIR / f"scores_{name}_{split}.npy", prob)
        np.save(config.RESULTS_DIR / f"contrib_{name}_{split}.npy", contrib)

        y = data["label"]
        if (y >= 0).all() and len(set(y)) == 2:
            print(f"\n{split}: ROC-AUC = {roc_auc_score(y, prob):.3f}, "
                  f"ακρίβεια (threshold 0.5) = {((prob > 0.5) == y).mean():.3f}")
            # ROC-AUC ανά τύπο θετικού ζεύγους (έναντι όλων των αρνητικών).
            for kind in PR.POSITIVE_TYPES:
                m = (data["type"] == kind) | (y == 0)
                if (data["type"] == kind).any():
                    print(f"   {kind:12s} ROC-AUC = {roc_auc_score(y[m], prob[m]):.3f}")
            # Μέση συνεισφορά κάθε κλάδου σε θετικά και αρνητικά ζεύγη.
            for label, title in ((1, "θετικά"), (0, "αρνητικά")):
                mean = contrib[y == label].mean(axis=0)
                parts = ", ".join(f"{b} {v:+.2f}" for b, v in zip(branches, mean))
                print(f"   μέση συνεισφορά ({title}): {parts}")
        elif (y == 0).all():
            # Σύνολο μόνο με ανεξάρτητα ζεύγη: ποσοστό ψευδών ενδείξεων.
            print(f"\n{split}: ψευδείς ενδείξεις (p > 0.5) = {(prob > 0.5).mean():.1%}, "
                  f"μέση βαθμολογία = {prob.mean():.3f}")
        else:
            print(f"{split}: αποθηκεύτηκαν {len(prob)} βαθμολογίες")


def main():
    parser = argparse.ArgumentParser(description="Εκπαίδευση Siamese ανιχνευτή")
    parser.add_argument("--quick", action="store_true",
                        help="γρήγορη δοκιμή (2000 ζεύγη, 2 εποχές)")
    parser.add_argument("--branches", default="melody,rhythm,harmony",
                        help="κλάδοι χωρισμένοι με κόμμα (π.χ. melody,rhythm)")
    parser.add_argument("--name", default="siamese", help="όνομα πειράματος")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--extra", nargs="*", default=[],
                        help="επιπλέον σύνολα αρνητικών (π.χ. remelodized)")
    parser.add_argument("--eval_only", action="store_true",
                        help="χωρίς εκπαίδευση: φόρτωση βαρών και αξιολόγηση")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help=f"seed εκπαίδευσης (προεπιλογή {config.SEED}). Με άλλη "
                             "τιμή το όνομα παίρνει το επίθημα _s<seed>")
    args = parser.parse_args()
    config.ensure_dirs()
    print(f"TensorFlow {tf.__version__}, Keras {keras.__version__}")

    # Με το προεπιλεγμένο seed το όνομα μένει ίδιο (π.χ. siamese_v2), ώστε τα
    # αρχικά αποτελέσματα να μην αλλάζουν. Οι επαναλήψεις αποθηκεύονται χωριστά
    # (π.χ. siamese_v2_s1) και δεν επηρεάζουν τους πίνακες του evaluate.py.
    base = args.name + (f"_s{args.seed}" if args.seed != config.SEED else "")

    if args.eval_only:
        import xai   # εδώ, για να αποφευχθεί κυκλική εισαγωγή
        _, explainer, branches = xai.load_trained(base)
        evaluate(explainer, base, branches, quick=False)
        return

    branches = tuple(b.strip() for b in args.branches.split(",") if b.strip())
    unknown = set(branches) - set(M.BRANCHES)
    if unknown:
        raise ValueError(f"Άγνωστοι κλάδοι: {unknown}. Επιτρεπτοί: {M.BRANCHES}")
    name = base + ("_quick" if args.quick else "")
    epochs = 2 if args.quick else args.epochs

    explainer = train(name, branches, epochs, args.quick, tuple(args.extra), args.seed)
    evaluate(explainer, name, branches, args.quick)


if __name__ == "__main__":
    main()
