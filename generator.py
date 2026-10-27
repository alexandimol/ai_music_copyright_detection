# -*- coding: utf-8 -*-
"""
generator.py
============
Μικρός generator μελωδιών (LSTM) για το πείραμα απομνημόνευσης.

Σκοπός δεν είναι η παραγωγή μουσικής υψηλής ποιότητας, αλλά η δημιουργία
ελεγχόμενης «AI-generated» μουσικής, της οποίας γνωρίζουμε ακριβώς τα
δεδομένα εκπαίδευσης. Έτσι μπορούμε να ελέγξουμε (memorization.py) αν ο
ανιχνευτής εντοπίζει αντιγραφή από το σύνολο εκπαίδευσης.

Εκπαιδεύονται δύο εκδοχές, ώστε να μελετηθεί η επίδραση του μεγέθους των
δεδομένων στην απομνημόνευση (πρβλ. Somepalli et al., 2022):
    gen_full  : όλα τα τμήματα εκπαίδευσης, με dropout (κανονική γενίκευση)
    gen_small : λίγα τμήματα (100), πολλά βήματα εκπαίδευσης, χωρίς dropout
                (σενάριο υπερπροσαρμογής — αναμένεται απομνημόνευση). Λειτουργεί
                ως «θετικός έλεγχος»: αν ο ανιχνευτής δεν βρίσκει αντιγραφή ούτε
                εδώ, τότε η απουσία ευρημάτων στο gen_full δεν θα σήμαινε τίποτα.

Αναπαράσταση (ένα σύμβολο ανά δέκατο έκτο, αυτοπαλίνδρομη μοντελοποίηση
όπως στους Briot, Hadjeres & Pachet, 2017):
    0      = παύση
    1      = συνέχιση της προηγούμενης νότας (hold)
    2..61  = έναρξη νότας με τονικό ύψος 36..95 (μεταφερμένο σε Ντο / Λα ελ.)
    62     = σύμβολο έναρξης ακολουθίας (BOS), μόνο ως είσοδος

Εκτέλεση:
    python generator.py --name gen_full
    python generator.py --name gen_small --subset 100 --epochs 400 --batch_size 16 --dropout 0
"""

import argparse
import json
import pickle
import random

import keras
import numpy as np
import pretty_midi
import tensorflow as tf
from keras import layers

import config
import preprocessing as P

T = config.SEGMENT_STEPS
REST, HOLD, NOTE0 = 0, 1, 2
N_PITCH = config.GEN_PITCH_HIGH - config.GEN_PITCH_LOW
VOCAB = NOTE0 + N_PITCH          # 62 σύμβολα εξόδου
BOS = VOCAB                      # 62: σύμβολο έναρξης (μόνο είσοδος)


# ===========================================================================
# 1. Μετατροπή τμήμα ↔ σύμβολα
# ===========================================================================
def key_shift(seg):
    """Μεταφορά (σε ημιτόνια) που φέρνει την τονικότητα του τμήματος σε Ντο/Λα ελ."""
    tonic, mode = P.estimate_key(seg["melody"], seg["accomp"])
    k = (-P.reference_pc(tonic, mode)) % 12
    return k - 12 if k > 6 else k


def segment_to_tokens(seg):
    """Μετατρέπει τη μελωδία ενός τμήματος σε ακολουθία T συμβόλων."""
    tokens = np.full(T, REST, dtype=np.int32)
    shift = key_shift(seg)
    for on, dur, pitch in seg["melody"]:
        p = int(pitch) + shift
        while p < config.GEN_PITCH_LOW:       # μεταφορά κατά οκτάβα αν
            p += 12                           # βγαίνει εκτός εύρους
        while p >= config.GEN_PITCH_HIGH:
            p -= 12
        end = min(T, on + dur)
        tokens[on] = NOTE0 + p - config.GEN_PITCH_LOW
        tokens[on + 1:end] = HOLD
    return tokens


def tokens_to_segment(tokens, seg_id, sec_per_step=0.125):
    """
    Μετατρέπει ακολουθία συμβόλων σε τμήμα (λίστα νοτών), στην ίδια μορφή
    με τα τμήματα του preprocessing.py, ώστε να συγκρίνεται με αυτά.
    Ένα «hold» χωρίς προηγούμενη νότα αντιμετωπίζεται ως παύση.
    """
    notes, current = [], None
    for t, tok in enumerate(tokens):
        tok = int(tok)
        if tok >= NOTE0:
            if current:
                notes.append(current)
            current = [t, 1, config.GEN_PITCH_LOW + tok - NOTE0]
        elif tok == HOLD and current:
            current[1] += 1
        else:
            if current:
                notes.append(current)
            current = None
    if current:
        notes.append(current)
    return {"song_id": f"GEN{seg_id:04d}", "seg_idx": 0, "start_step": 0,
            "melody": np.array(notes, dtype=np.int16).reshape(-1, 3),
            "accomp": np.zeros((0, 3), dtype=np.int16),
            "sec_per_step": sec_per_step}


# ===========================================================================
# 2. Μοντέλο
# ===========================================================================
def build_generator(dropout):
    """Embedding → LSTM → Dense: πρόβλεψη του επόμενου συμβόλου."""
    inputs = keras.Input(shape=(None,), dtype="int32", name="tokens")
    x = layers.Embedding(VOCAB + 1, config.GEN_EMBED_DIM)(inputs)
    x = layers.LSTM(config.GEN_LSTM_UNITS, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    logits = layers.Dense(VOCAB)(x)
    return keras.Model(inputs, logits, name="melody_lstm")


def make_xy(tokens):
    """Είσοδος = [BOS] + σύμβολα[:-1], στόχος = σύμβολα (teacher forcing)."""
    bos = np.full((len(tokens), 1), BOS, dtype=np.int32)
    return np.concatenate([bos, tokens[:, :-1]], axis=1), tokens


def sample(model, n, temperature, seed):
    """
    Αυτοπαλίνδρομη δειγματοληψία n μελωδιών μήκους T. Σε κάθε βήμα το
    μοντέλο δίνει κατανομή για το επόμενο σύμβολο και επιλέγεται τυχαία ένα
    σύμβολο (θερμοκρασία: <1 πιο «ασφαλείς», >1 πιο τολμηρές επιλογές).
    """
    tf.random.set_seed(seed)
    seq = np.full((n, 1), BOS, dtype=np.int32)
    for _ in range(T):
        logits = model(seq, training=False)[:, -1, :] / temperature
        nxt = tf.random.categorical(logits, 1, dtype=tf.int32).numpy()
        seq = np.concatenate([seq, nxt], axis=1)
    return seq[:, 1:]


def export_midi(seg, path):
    """Αποθηκεύει μια μελωδία σε MIDI για ακρόαση."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, name="Melody")
    step = seg["sec_per_step"]
    for on, dur, pitch in seg["melody"]:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(pitch),
                                           start=on * step, end=(on + dur) * step))
    pm.instruments.append(inst)
    pm.write(str(path))


# ===========================================================================
# 3. Κύρια διαδικασία
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Εκπαίδευση generator μελωδιών")
    parser.add_argument("--name", default="gen_full")
    parser.add_argument("--subset", type=int, default=0,
                        help="πλήθος τμημάτων εκπαίδευσης (0 = όλα)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=config.GEN_BATCH_SIZE)
    parser.add_argument("--quick", action="store_true", help="γρήγορη δοκιμή")
    args = parser.parse_args()
    config.ensure_dirs()
    keras.utils.set_random_seed(config.SEED)

    segments, splits = P.load_segments()
    train_ids = set(splits["train"])
    train_segs = [s for s in segments if s["song_id"] in train_ids]
    if args.subset:
        train_segs = random.Random(config.SEED).sample(train_segs, args.subset)
    epochs, n_samples = args.epochs, config.GEN_N_SAMPLES
    if args.quick:
        train_segs, epochs, n_samples = train_segs[:300], 1, 10
    print(f"Generator '{args.name}': {len(train_segs)} τμήματα, {epochs} εποχές, "
          f"dropout {args.dropout}")

    tokens = np.stack([segment_to_tokens(s) for s in train_segs])
    x, y = make_xy(tokens)
    model = build_generator(args.dropout)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    history = model.fit(x, y, batch_size=args.batch_size, epochs=epochs,
                        shuffle=True, verbose=2)
    final_loss = float(history.history["loss"][-1])
    print(f"Τελικό σφάλμα εκπαίδευσης (loss): {final_loss:.3f}")
    if args.subset and final_loss > 0.3:
        print("Προσοχή: ο generator ΔΕΝ υπερπροσαρμόστηκε (loss > 0.3). Για σενάριο "
              "απομνημόνευσης αυξήστε τις εποχές ή μειώστε το --batch_size.")
    model.save_weights(str(config.MODELS_DIR / f"{args.name}.weights.h5"))

    # Παραγωγή νέων μελωδιών.
    generated = sample(model, n_samples, config.GEN_TEMPERATURE, config.SEED)
    gen_segs = [tokens_to_segment(g, i) for i, g in enumerate(generated)]
    gen_segs = [s for s in gen_segs if len(s["melody"]) >= 4]   # όχι «άδειες» μελωδίες

    out = config.GENERATED_DIR
    with open(out / f"{args.name}.pkl", "wb") as f:
        pickle.dump(gen_segs, f)
    with open(out / f"{args.name}_train_ids.json", "w", encoding="utf-8") as f:
        json.dump({"train": [[s["song_id"], s["seg_idx"]] for s in train_segs],
                   "final_loss": float(history.history["loss"][-1]),
                   "final_accuracy": float(history.history["accuracy"][-1])}, f)
    midi_dir = out / f"{args.name}_midi"
    midi_dir.mkdir(exist_ok=True)
    for s in gen_segs[:10]:
        export_midi(s, midi_dir / f"{s['song_id']}.mid")

    # Σύντομα στατιστικά (έλεγχος ότι οι μελωδίες είναι εύλογες).
    def stats(segs):
        n_notes = np.mean([len(s["melody"]) for s in segs])
        span = np.mean([np.ptp(s["melody"][:, 2]) for s in segs if len(s["melody"])])
        return f"{n_notes:.1f} νότες/τμήμα, μέσο εύρος {span:.1f} ημιτόνια"
    print(f"Δεδομένα εκπαίδευσης: {stats(train_segs)}")
    print(f"Παραγόμενες ({len(gen_segs)}): {stats(gen_segs)}")
    print(f"Αποθηκεύτηκαν στο {out}")


if __name__ == "__main__":
    main()
