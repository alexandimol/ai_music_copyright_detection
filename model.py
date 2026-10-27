# -*- coding: utf-8 -*-
"""
model.py
========
Siamese δίκτυο με πίνακες ομοιότητας (cross-similarity matrices) για την
ανίχνευση μουσικής αντιγραφής.

Ιδέα
----
1. Κάθε μουσική διάσταση (μελωδία, ρυθμός, αρμονία) έχει τον δικό της
   κωδικοποιητή (encoder) 1D-CNN. Ο ΙΔΙΟΣ κωδικοποιητής (ίδια βάρη)
   εφαρμόζεται και στα δύο τμήματα του ζεύγους — αυτό είναι το «Siamese»
   χαρακτηριστικό (Bromley et al., 1993; Chopra, Hadsell & LeCun, 2005).
2. Ο κωδικοποιητής παράγει ένα διάνυσμα ανά χρόνο (32 χρόνοι ανά τμήμα).
   Από τα διανύσματα των δύο τμημάτων υπολογίζεται ο πίνακας ομοιότητας
   συνημιτόνου S (32 × 32): το S[i, j] δείχνει πόσο μοιάζει ο χρόνος i του
   Α με τον χρόνο j του Β. Ένα αντιγραμμένο απόσπασμα εμφανίζεται ως
   ΔΙΑΓΩΝΙΑ γραμμή υψηλών τιμών, ανεξάρτητα από τη θέση του — η ίδια αρχή
   χρησιμοποιείται στην αναγνώριση διασκευών (Serrà, Serra & Andrzejak, 2009).
3. Ένα μικρό 2D-CNN ανά διάσταση «διαβάζει» τον πίνακα και παράγει μια
   συνεισφορά (logit) c_k. Η τελική βαθμολογία ομοιότητας είναι
        p = σ(c_μελωδία + c_ρυθμός + c_αρμονία)
   Η p ερμηνεύεται ως βαθμολογία στο [0, 1] και όχι ως πιθανότητα αντιγραφής
   (το δίκτυο εκπαιδεύεται σε ισορροπημένα συνθετικά ζεύγη).
   Η ΠΡΟΣΘΕΤΙΚΗ δομή είναι σκόπιμη επιλογή ερμηνευσιμότητας: η συμβολή κάθε
   διάστασης στην απόφαση διαβάζεται απευθείας, χωρίς προσεγγίσεις.

Έτσι το μοντέλο εξηγεί τι είδε με δύο τρόπους: οι πίνακες S δείχνουν ΠΟΥ
εντοπίζεται η ομοιότητα, οι συνεισφορές c_k δείχνουν ΣΕ ΤΙ (μελωδία, ρυθμός,
αρμονία).
"""

import keras
import tensorflow as tf
from keras import layers

import config

T = config.SEGMENT_STEPS
BRANCHES = ("melody", "rhythm", "harmony")

# Διάσταση εισόδου κάθε κλάδου (κανάλια ανά χρονικό βήμα):
#   melody  = one-hot τονικού ύψους (50) + one-hot διαστήματος (26)
#   rhythm  = [έναρξη νότας, νότα σε διάρκεια, πυκνότητα ενάρξεων συνοδείας]
#   harmony = chroma συνοδείας (12)
INPUT_DIMS = {
    "melody": config.N_PITCH_CLASSES + config.N_INTERVAL_CLASSES,
    "rhythm": 3,
    "harmony": 12,
}


# ===========================================================================
# 1. Προετοιμασία εισόδων (από τα αποθηκευμένα ζεύγη)
# ===========================================================================
def to_model_inputs(batch):
    """
    Μετατρέπει ένα λεξικό με τα αποθηκευμένα πεδία (δείκτες κλάσεων) στις
    εισόδους του μοντέλου (one-hot, float32). Λειτουργεί τόσο σε tf.Tensor
    (μέσα σε tf.data) όσο και σε πίνακες numpy.
    """
    out = {}
    for side in ("a", "b"):
        pitch = tf.one_hot(batch[f"{side}_melody"], config.N_PITCH_CLASSES)
        interval = tf.one_hot(batch[f"{side}_interval"], config.N_INTERVAL_CLASSES)
        out[f"{side}_melody"] = tf.concat([pitch, interval], axis=-1)
        out[f"{side}_rhythm"] = tf.cast(batch[f"{side}_rhythm"], tf.float32)
        out[f"{side}_harmony"] = tf.cast(batch[f"{side}_harmony"], tf.float32)
    return out


def swap_sides(inputs):
    """Ανταλλάσσει τα Α και Β (επαύξηση δεδομένων: η ομοιότητα είναι συμμετρική)."""
    return {("b" + k[1:] if k.startswith("a_") else "a" + k[1:]): v
            for k, v in inputs.items()}


# ===========================================================================
# 2. Δομικά στοιχεία
# ===========================================================================
def make_encoder(branch, filters=64, emb_dim=config.EMBEDDING_DIM, dropout=0.1):
    """
    Κωδικοποιητής 1D-CNN για έναν κλάδο.
    Είσοδος (T=128, κανάλια) → έξοδος (32, emb_dim): ένα διάνυσμα ανά χρόνο,
    κανονικοποιημένο σε μοναδιαίο μήκος (ώστε το εσωτερικό γινόμενο να είναι
    ομοιότητα συνημιτόνου).
    """
    return keras.Sequential([
        keras.Input(shape=(T, INPUT_DIMS[branch])),
        layers.Conv1D(filters, 5, padding="same", activation="relu"),
        layers.Conv1D(filters, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),                     # 128 → 64 βήματα
        layers.Dropout(dropout),
        layers.Conv1D(filters, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),                     # 64 → 32 (ένα ανά χρόνο)
        layers.Conv1D(emb_dim, 3, padding="same"),
        layers.UnitNormalization(axis=-1),
    ], name=f"encoder_{branch}")


def make_head(branch, dropout=0.2):
    """
    2D-CNN που «διαβάζει» τον πίνακα ομοιότητας (32 × 32) και επιστρέφει μία
    τιμή (logit): τη συνεισφορά του κλάδου στην απόφαση. Το GlobalMaxPooling
    κάνει την ανίχνευση ανεξάρτητη από τη ΘΕΣΗ του αντιγραμμένου αποσπάσματος.
    """
    n = config.SEGMENT_BEATS
    return keras.Sequential([
        keras.Input(shape=(n, n, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.GlobalMaxPooling2D(),
        layers.Dropout(dropout),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ], name=f"head_{branch}")


# ===========================================================================
# 3. Κατασκευή μοντέλου
# ===========================================================================
def build_models(branches=BRANCHES):
    """
    Κατασκευάζει δύο μοντέλα με ΚΟΙΝΑ βάρη:
        detector  : εισόδους → βαθμολογία ομοιότητας p (για εκπαίδευση/αξιολόγηση)
        explainer : εισόδους → {prob, contrib, sim_<κλάδος>} (για το XAI)

    branches: ποιοι κλάδοι χρησιμοποιούνται (για πειράματα αφαίρεσης - ablation).
    """
    branches = tuple(branches)
    inputs, contribs, sims = {}, [], {}
    for branch in branches:
        a = keras.Input(shape=(T, INPUT_DIMS[branch]), name=f"a_{branch}")
        b = keras.Input(shape=(T, INPUT_DIMS[branch]), name=f"b_{branch}")
        inputs[f"a_{branch}"], inputs[f"b_{branch}"] = a, b

        encoder = make_encoder(branch)
        ea, eb = encoder(a), encoder(b)             # ίδια βάρη → Siamese
        # Πίνακας ομοιότητας συνημιτόνου: (32, emb) · (32, emb)ᵀ → (32, 32)
        sim = layers.Dot(axes=(2, 2), name=f"sim_{branch}")([ea, eb])
        sims[branch] = sim
        sim_img = layers.Reshape((config.SEGMENT_BEATS, config.SEGMENT_BEATS, 1))(sim)
        contribs.append(make_head(branch)(sim_img))

    logit = contribs[0] if len(contribs) == 1 else layers.Add(name="logit")(contribs)
    prob = layers.Activation("sigmoid", name="prob")(logit)
    contrib = (contribs[0] if len(contribs) == 1
               else layers.Concatenate(name="contrib")(contribs))

    detector = keras.Model(inputs, prob, name="siamese_detector")
    explainer = keras.Model(
        inputs, {"prob": prob, "contrib": contrib,
                 **{f"sim_{k}": v for k, v in sims.items()}},
        name="siamese_explainer")
    return detector, explainer


def select_inputs(inputs, branches):
    """Κρατά μόνο τις εισόδους των κλάδων που χρησιμοποιεί το μοντέλο."""
    return {k: v for k, v in inputs.items() if k.split("_", 1)[1] in branches}


if __name__ == "__main__":
    # Εκτύπωση της αρχιτεκτονικής (χρήσιμο για την ενότητα Μεθοδολογίας).
    det, _ = build_models()
    det.summary(expand_nested=True)
