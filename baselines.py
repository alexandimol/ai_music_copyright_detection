# -*- coding: utf-8 -*-
"""
baselines.py
============
Κλασικές (μη νευρωνικές) μέθοδοι μέτρησης μουσικής ομοιότητας, που
χρησιμοποιούνται ως σημεία αναφοράς (baselines) για τη σύγκριση με το
νευρωνικό μοντέλο.

Συμβολικό επίπεδο:
    1. interval_match   : ποσοστό κοινών μελωδικών διαστημάτων ανά χρονικό βήμα,
                          με ανοχή σε μικρή χρονική μετατόπιση (±1 χρόνος).
    2. smith_waterman   : τοπική στοίχιση (local alignment) των ακολουθιών
                          διαστημάτων σε επίπεδο νότας (Smith & Waterman, 1981).
                          Εντοπίζει το πιο όμοιο κοινό απόσπασμα, ακόμη κι αν
                          βρίσκεται σε διαφορετική θέση — κατάλληλο για μερική
                          αντιγραφή. Η αναπαράσταση της μελωδίας ως ακολουθίας
                          διαστημάτων χρησιμοποιείται ευρέως σε αλγορίθμους
                          μελωδικής ομοιότητας (Müllensiefen & Pendzich, 2009).
    3. harmony_cosine   : ομοιότητα συνημιτόνου της αρμονίας (chroma συνοδείας).
                          Μετρά μόνο «ύφος/αρμονία» και όχι μελωδία.

Ηχητικό επίπεδο:
    4. audio_chroma_dtw : σύνθεση ήχου από το MIDI, εξαγωγή χαρακτηριστικών chroma
                          ανά χρόνο (Müller, 2015), ευθυγράμμιση τονικότητας με τη
                          μέθοδο του βέλτιστου δείκτη μεταφοράς και σύγκριση με
                          Dynamic Time Warping (Sakoe & Chiba, 1978).

Κάθε μέθοδος επιστρέφει βαθμολογία ομοιότητας (μεγαλύτερη = πιο όμοια).
Οι βαθμολογίες αποθηκεύονται στο outputs/results/ ώστε το evaluate.py να τις
συγκρίνει με το νευρωνικό μοντέλο.

Εκτέλεση:
    python baselines.py            (όλες οι μέθοδοι, ~15–20 λεπτά σε CPU)
    python baselines.py --quick    (γρήγορη δοκιμή σε 200 ζεύγη)
    python baselines.py --splits test_remelodized   (μόνο τα αρνητικά της v2)
"""

import argparse
import time

import numpy as np
from scipy.signal import stft
from sklearn.metrics import roc_auc_score

import config
import pairs as PR

T = config.SEGMENT_STEPS
SPLITS = ["val", "test", "sweep_melodic", "sweep_partial", "test_remelodized"]


# ===========================================================================
# 1. Αντιστοίχιση διαστημάτων ανά χρονικό βήμα
# ===========================================================================
def interval_match(data, max_shift=config.STEPS_PER_BEAT):
    """
    Για κάθε ζεύγος υπολογίζει το ποσοστό χρονικών βημάτων όπου τα δύο
    τμήματα έχουν το ίδιο μελωδικό διάστημα (δείκτης Jaccard), και κρατά
    το μέγιστο για χρονικές μετατοπίσεις έως ±max_shift βήματα.
    """
    a, b = data["a_interval"], data["b_interval"]
    best = np.zeros(len(a))
    for s in range(-max_shift, max_shift + 1):
        bs = np.roll(b, s, axis=1)
        inter = ((a == bs) & (a > 0)).sum(axis=1)
        union = ((a > 0) | (bs > 0)).sum(axis=1)
        best = np.maximum(best, inter / np.maximum(union, 1))
    return best


# ===========================================================================
# 2. Τοπική στοίχιση Smith–Waterman σε ακολουθίες διαστημάτων
# ===========================================================================
def note_intervals(melody):
    """Ακολουθία μελωδικών διαστημάτων (σε ημιτόνια) μεταξύ διαδοχικών νοτών."""
    return np.diff(melody[:, 2].astype(int)).tolist()


def smith_waterman(x, y, match=2.0, near=0.5, mismatch=-1.0, gap=-1.0):
    """
    Αλγόριθμος Smith–Waterman για τοπική στοίχιση δύο ακολουθιών.

    Βαθμολόγηση:
        ίδιο διάστημα                 : +match
        διάστημα που διαφέρει κατά 1  : +near  (π.χ. μεγάλη/μικρή τρίτη)
        διαφορετικό διάστημα          : mismatch
        κενό (παράλειψη/προσθήκη νότας): gap

    Επιστρέφει τη μέγιστη βαθμολογία στοίχισης, κανονικοποιημένη ως προς τη
    μέγιστη δυνατή (match × μήκος της μικρότερης ακολουθίας), στο [0, 1].
    """
    if not x or not y:
        return 0.0
    prev = [0.0] * (len(y) + 1)
    best = 0.0
    for xi in x:
        cur = [0.0] * (len(y) + 1)
        for j, yj in enumerate(y, 1):
            d = abs(xi - yj)
            s = match if d == 0 else (near if d == 1 else mismatch)
            v = max(0.0, prev[j - 1] + s, prev[j] + gap, cur[j - 1] + gap)
            cur[j] = v
            if v > best:
                best = v
        prev = cur
    return best / (match * min(len(x), len(y)))


def smith_waterman_scores(pair_notes):
    """Βαθμολογία Smith–Waterman για κάθε ζεύγος (σε επίπεδο νότας)."""
    return np.array([smith_waterman(note_intervals(p["a"]["melody"]),
                                    note_intervals(p["b"]["melody"]))
                     for p in pair_notes])


# ===========================================================================
# 3. Ομοιότητα αρμονίας
# ===========================================================================
def harmony_cosine(data):
    """Ομοιότητα συνημιτόνου των chroma της συνοδείας (ανά μέτρο)."""
    def per_bar(h):
        n = len(h)
        return h.reshape(n, T // config.STEPS_PER_BAR, -1, 12).mean(axis=2).reshape(n, -1)
    a, b = per_bar(data["a_harmony"]), per_bar(data["b_harmony"])
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return num / den


# ===========================================================================
# 4. Ηχητικό επίπεδο: σύνθεση → chroma → DTW
# ===========================================================================
SR = 8000           # συχνότητα δειγματοληψίας για τη σύνθεση (αρκεί έως 4 kHz)
N_FFT = 4096        # μήκος παραθύρου STFT (~0.5 s, ανάλυση ~2 Hz)
HOP = 512


def synthesize(seg, sr=SR):
    """
    Απλή σύνθεση ήχου από τις νότες ενός τμήματος (ημιτονοειδή κύματα με
    αρμονικές και εκθετική απόσβεση). Η μελωδία αποδίδεται με μεγαλύτερη
    ένταση από τη συνοδεία. Επιστρέφει (σήμα, διάρκεια_χρόνου_σε_δείγματα).
    """
    step = seg["sec_per_step"]
    n = int(np.ceil(T * step * sr)) + sr // 4
    y = np.zeros(n, dtype=np.float32)
    for notes, gain in ((seg["melody"], 1.0), (seg["accomp"], 0.35)):
        for on, dur, pitch in notes:
            start = int(on * step * sr)
            length = max(1, int(dur * step * sr))
            t = np.arange(length) / sr
            f0 = 440.0 * 2 ** ((int(pitch) - 69) / 12)
            env = np.exp(-3.0 * t)
            tone = (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(4 * np.pi * f0 * t)
                    + 0.25 * np.sin(6 * np.pi * f0 * t))
            end = min(n, start + length)
            y[start:end] += gain * (env * tone)[:end - start]
    return y, step * config.STEPS_PER_BEAT * sr


# Αντιστοίχιση κάθε συχνοτικού bin του STFT σε τονική κλάση (55 Hz – 4 kHz).
_FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)
_VALID = (_FREQS >= 55) & (_FREQS < SR / 2)
_PC = np.zeros(len(_FREQS), dtype=int)
_PC[_VALID] = np.round(12 * np.log2(_FREQS[_VALID] / 440.0) + 69).astype(int) % 12


def audio_chroma(seg):
    """
    Χαρακτηριστικά chroma από τον συνθεμένο ήχο, με ένα διάνυσμα ανά χρόνο
    (beat-synchronous chroma, βλ. Müller, 2015). Κάθε διάνυσμα κανονικοποιείται.
    """
    y, beat_len = synthesize(seg)
    _, times, Z = stft(y, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    mag = np.log1p(100 * np.abs(Z))            # λογαριθμική συμπίεση
    chroma = np.zeros((12, mag.shape[1]))
    for pc in range(12):
        chroma[pc] = mag[_VALID & (_PC == pc)].sum(axis=0)
    beat_idx = np.minimum((times * SR / beat_len).astype(int), config.SEGMENT_BEATS - 1)
    out = np.zeros((config.SEGMENT_BEATS, 12))
    for b in range(config.SEGMENT_BEATS):
        cols = chroma[:, beat_idx == b]
        if cols.size:
            out[b] = cols.mean(axis=1)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)


def dtw_cost(A, B):
    """
    Dynamic Time Warping (Sakoe & Chiba, 1978) με απόσταση συνημιτόνου.
    Επιστρέφει το κανονικοποιημένο κόστος της βέλτιστης διαδρομής.
    """
    C = 1.0 - A @ B.T
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return D[n, m] / (n + m)


def audio_chroma_dtw(pair_notes):
    """
    Ηχητικό baseline. Πριν από το DTW, το chroma του δεύτερου τμήματος
    περιστρέφεται κατά τον «βέλτιστο δείκτη μεταφοράς» (optimal transposition
    index): τη μετατόπιση που μεγιστοποιεί την ομοιότητα των μέσων chroma.
    Έτσι η σύγκριση δεν επηρεάζεται από αλλαγή τονικότητας.
    """
    cache = {}   # το ίδιο τμήμα αναφοράς εμφανίζεται σε πολλά ζεύγη

    def chroma_of(seg):
        if id(seg) not in cache:
            cache[id(seg)] = audio_chroma(seg)
        return cache[id(seg)]

    scores = []
    for p in pair_notes:
        A, B = chroma_of(p["a"]), chroma_of(p["b"])
        ga, gb = A.mean(axis=0), B.mean(axis=0)
        k = int(np.argmax([ga @ np.roll(gb, s) for s in range(12)]))
        scores.append(-dtw_cost(A, np.roll(B, k, axis=1)))  # μικρό κόστος = όμοια
    return np.array(scores)


# ===========================================================================
# 5. Εκτέλεση και αποθήκευση
# ===========================================================================
METHODS = ["interval_match", "smith_waterman", "harmony_cosine", "audio_chroma_dtw"]


def run_method(name, data, pair_notes):
    if name == "interval_match":
        return interval_match(data)
    if name == "harmony_cosine":
        return harmony_cosine(data)
    if name == "smith_waterman":
        return smith_waterman_scores(pair_notes)
    if name == "audio_chroma_dtw":
        return audio_chroma_dtw(pair_notes)
    raise ValueError(name)


def subset(data, pair_notes, n):
    """Επιλέγει τα πρώτα n ζεύγη (για γρήγορη δοκιμή)."""
    return {k: v[:n] for k, v in data.items()}, pair_notes[:n]


def main():
    parser = argparse.ArgumentParser(description="Κλασικές μέθοδοι ομοιότητας")
    parser.add_argument("--quick", action="store_true",
                        help="γρήγορη δοκιμή σε 200 ζεύγη ανά σύνολο")
    parser.add_argument("--splits", nargs="*", default=SPLITS,
                        help="ποια σύνολα (π.χ. --splits test_remelodized)")
    args = parser.parse_args()
    config.ensure_dirs()

    for split in args.splits:
        if not (config.PAIRS_DIR / f"{split}.npz").exists():
            print(f"Παράλειψη {split}: δεν υπάρχει (τρέξτε pairs.py)")
            continue
        data, notes = PR.load_pairs(split), PR.load_pair_notes(split)
        if args.quick:
            data, notes = subset(data, notes, 200)
        print(f"\n=== {split} ({len(notes)} ζεύγη) ===")
        for name in METHODS:
            t0 = time.time()
            scores = run_method(name, data, notes)
            np.save(config.RESULTS_DIR / f"scores_{name}_{split}.npy", scores)
            msg = f"  {name:18s} {time.time() - t0:6.1f} s"
            y = data["label"]
            if (y >= 0).all() and len(set(y)) == 2:
                msg += f"   ROC-AUC = {roc_auc_score(y, scores):.3f}"
            print(msg)


if __name__ == "__main__":
    main()
