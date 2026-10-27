# -*- coding: utf-8 -*-
"""
pairs.py
========
Δημιουργία επισημασμένων ζευγών (labelled pairs) για την εκπαίδευση και
αξιολόγηση του ανιχνευτή ομοιότητας.

Πρόβλημα: δεν υπάρχει δημόσιο σύνολο δεδομένων με πραγματικές υποθέσεις
παραβίασης πνευματικών δικαιωμάτων σε αρκετό όγκο για εκπαίδευση
νευρωνικού δικτύου. Λύση: κατασκευάζουμε ζεύγη εφαρμόζοντας ΕΛΕΓΧΟΜΕΝΟΥΣ
μετασχηματισμούς σε τμήματα του POP909, ώστε να γνωρίζουμε με βεβαιότητα
αν (και σε ποιο βαθμό) το δεύτερο τμήμα προέρχεται από το πρώτο.

Θετικά ζεύγη (label = 1, «ουσιώδης αντιγραφή της μελωδίας»):
    transposed    : αντίγραφο σε άλλη τονικότητα
    melodic       : αλλαγή τονικού ύψους σε 5–30% των νοτών
    rhythmic      : ρυθμική αλλοίωση σε 5–30% των νοτών
    ornamented    : προσθήκη στολιδιών (διαβατικές/γειτονικές νότες) και
                    μετατόπιση ορισμένων νοτών κατά οκτάβα
    combined      : συνδυασμός μελωδικής, ρυθμικής αλλοίωσης και στολιδιών
    partial       : αντιγραφή μέρους της μελωδίας (25–75%, π.χ. ένα «hook»),
                    τοποθετημένου σε ΤΥΧΑΙΑ θέση μέσα σε άλλο τραγούδι
    new_accomp    : ίδια μελωδία με εντελώς διαφορετική συνοδεία
Σε όλα τα θετικά εφαρμόζεται επιπλέον, με πιθανότητα 50%, μικρή χρονική
μετατόπιση (έως έναν χρόνο), ώστε το μοντέλο να μη βασίζεται σε τέλεια
χρονική ευθυγράμμιση.

Αρνητικά ζεύγη (label = 0, «ανεξάρτητη δημιουργία»):
    random        : τμήμα από άλλο τραγούδι
    hard_harmony  : τμήμα άλλου τραγουδιού με την ΠΙΟ ΠΑΡΟΜΟΙΑ αρμονική πορεία
                    (π.χ. ίδια αλληλουχία συγχορδιών) αλλά διαφορετική μελωδία
    hard_melody   : τμήμα άλλου τραγουδιού με το ΠΙΟ ΠΑΡΟΜΟΙΟ μελωδικό προφίλ
                    (κατανομή διαστημάτων και ρυθμού), δηλαδή κοινά
                    «δομικά στοιχεία» της ποπ μουσικής χωρίς αντιγραφή
Τα δύο «δύσκολα» αρνητικά αντιστοιχούν στη νομική διάκριση ανάμεσα σε κοινά
υφολογικά ή αρμονικά στοιχεία, που δεν προστατεύονται αυτοτελώς, και στην
αντιγραφή πρωτότυπης έκφρασης.

Έκδοση v2 (επιπλέον σύνολα *_remelodized, python pairs.py --extra):
    remelodized   : ΙΔΙΟ τμήμα με τον ίδιο ρυθμό και τις ίδιες θέσεις νοτών,
                    αλλά με 70–100% των τονικών υψών αλλαγμένα. Η αξιολόγηση
                    της v1 έδειξε ότι το μοντέλο θεωρούσε «αντιγραφή» κάθε
                    τμήμα με τον ίδιο ρυθμικό σκελετό· τα ζεύγη αυτά διδάσκουν
                    ότι προστατεύεται η μελωδική έκφραση και όχι ο ρυθμός μόνος.

Για να μη «μάθει» το μοντέλο απλώς να αναγνωρίζει ίχνη επεξεργασίας (και όχι
την ομοιότητα), εφαρμόζουμε ελαφριά αλλοίωση και στα μισά αρνητικά ζεύγη.

Εκτέλεση:
    python pairs.py
"""

import argparse
import pickle
import random
from collections import defaultdict

import numpy as np

import config
import preprocessing as P

T = config.SEGMENT_STEPS
BAR = config.STEPS_PER_BAR


# ===========================================================================
# 1. Βασικοί μετασχηματισμοί (σε επίπεδο νότας)
# ===========================================================================
def copy_segment(seg):
    """Αντίγραφο τμήματος ώστε οι μετασχηματισμοί να μην αλλοιώνουν το αρχικό."""
    new = dict(seg)
    new["melody"] = seg["melody"].astype(np.int32).copy()
    new["accomp"] = seg["accomp"].astype(np.int32).copy()
    return new


def transpose(seg, k):
    """Μεταφορά ολόκληρου του τμήματος (μελωδία + συνοδεία) κατά k ημιτόνια."""
    new = copy_segment(seg)
    new["melody"][:, 2] += k
    new["accomp"][:, 2] += k
    return new


def random_transpose(seg, rng):
    """Μεταφορά κατά τυχαίο πλήθος ημιτονίων στο [-6, 6] (εκτός του 0)."""
    k = rng.choice([i for i in range(-config.MAX_TRANSPOSE,
                                     config.MAX_TRANSPOSE + 1) if i != 0])
    return transpose(seg, k)


def fix_notes(notes, monophonic):
    """
    Επαναφέρει μια λίστα νοτών σε έγκυρη μορφή μετά από αλλοίωση:
    απόρριψη νοτών εκτός τμήματος, περικοπή διάρκειας, ταξινόμηση και
    (για τη μελωδία) μονοφωνία.
    """
    notes = np.asarray(notes, dtype=np.int32).reshape(-1, 3)
    notes = notes[(notes[:, 0] >= 0) & (notes[:, 0] < T)]
    notes[:, 1] = np.clip(notes[:, 1], 1, T - notes[:, 0])
    notes = notes[np.lexsort((notes[:, 2], notes[:, 0]))]
    return P.make_monophonic(notes) if monophonic else notes


def time_shift(seg, s):
    """
    Χρονική μετατόπιση ολόκληρου του τμήματος κατά s βήματα. Οι νότες που
    βγαίνουν εκτός τμήματος απορρίπτονται (όπως όταν ένα αντιγραμμένο
    απόσπασμα εμφανίζεται λίγο νωρίτερα ή αργότερα στο νέο έργο).
    """
    new = copy_segment(seg)
    new["melody"][:, 0] += s
    new["accomp"][:, 0] += s
    new["melody"] = fix_notes(new["melody"], monophonic=True)
    new["accomp"] = fix_notes(new["accomp"], monophonic=False)
    return new


def scale_neighbours(pitch, scale, max_dist=5):
    """Νότες της κλίμακας σε απόσταση έως max_dist ημιτονίων (εκτός της ίδιας)."""
    return [pitch + d for d in range(-max_dist, max_dist + 1)
            if d != 0 and (pitch + d) % 12 in scale]


def melodic_variation(seg, p, rng):
    """
    Αλλάζει το τονικό ύψος σε ποσοστό p των νοτών της μελωδίας.

    Η νέα νότα επιλέγεται τυχαία από τις νότες της ΙΔΙΑΣ κλίμακας σε απόσταση
    έως 5 ημιτονίων, ώστε το αποτέλεσμα να παραμένει μουσικά εύλογο (και όχι
    τυχαίος θόρυβος), όπως θα έκανε κάποιος που «παραλλάσσει» μια μελωδία.
    """
    new = copy_segment(seg)
    mel = new["melody"]
    n_change = int(round(p * len(mel)))
    if n_change == 0:
        return new
    scale = P.scale_of(*P.estimate_key(seg["melody"], seg["accomp"]))
    for i in rng.sample(range(len(mel)), n_change):
        old = int(mel[i, 2])
        cands = scale_neighbours(old, scale)
        mel[i, 2] = rng.choice(cands) if cands else old + rng.choice([-1, 1])
    return new


def rhythmic_variation(seg, p, rng):
    """
    Ρυθμική αλλοίωση σε ποσοστό p των νοτών της μελωδίας. Για κάθε
    επιλεγμένη νότα εφαρμόζεται τυχαία μία από τις πράξεις:
        shift  : μετατόπιση της έναρξης κατά ±1 ή ±2 βήματα (συγκοπή)
        length : αλλαγή διάρκειας (±1 ή ±2 βήματα)
        split  : διαχωρισμός της νότας σε δύο ίσες νότες ίδιου ύψους
    """
    new = copy_segment(seg)
    mel = [list(n) for n in new["melody"]]
    extra = []
    for i in rng.sample(range(len(mel)), int(round(p * len(mel)))):
        op = rng.choice(["shift", "length", "split"])
        on, dur, pitch = mel[i]
        if op == "shift":
            mel[i][0] = on + rng.choice([-2, -1, 1, 2])
        elif op == "length":
            mel[i][1] = max(1, dur + rng.choice([-2, -1, 1, 2]))
        elif dur >= 2:   # split
            half = dur // 2
            mel[i][1] = half
            extra.append([on + half, dur - half, pitch])
    new["melody"] = fix_notes(mel + extra, monophonic=True)
    return new


def ornamentation(seg, p, p_octave, rng):
    """
    Προσθήκη στολιδιών σε ποσοστό p των νοτών: το δεύτερο μισό μιας νότας
    αντικαθίσταται από γειτονική νότα της κλίμακας (γειτονική ή διαβατική
    νότα — neighbour/passing note). Επιπλέον, ποσοστό p_octave των νοτών
    μετατοπίζεται κατά ±1 οκτάβα. Και οι δύο πράξεις αλλάζουν τα μελωδικά
    διαστήματα, αλλά διατηρούν τον αναγνωρίσιμο «σκελετό» της μελωδίας.
    """
    new = copy_segment(seg)
    scale = P.scale_of(*P.estimate_key(seg["melody"], seg["accomp"]))
    mel = [list(n) for n in new["melody"]]
    extra = []
    for i in rng.sample(range(len(mel)), int(round(p * len(mel)))):
        on, dur, pitch = mel[i]
        if dur < 2:
            continue
        half = dur // 2
        mel[i][1] = half
        cands = scale_neighbours(pitch, scale, max_dist=2)
        extra.append([on + half, dur - half,
                      rng.choice(cands) if cands else pitch + 1])
    mel = mel + extra
    for i in rng.sample(range(len(mel)), int(round(p_octave * len(mel)))):
        mel[i][2] += rng.choice([-12, 12])
    new["melody"] = fix_notes(mel, monophonic=True)
    return new


def align_to(other, anchor):
    """
    Μεταφέρει το τμήμα other ώστε η τονικότητά του να ταιριάζει με του anchor.
    Χρησιμοποιείται όταν συνδυάζουμε υλικό από δύο τραγούδια, ώστε το
    αποτέλεσμα να είναι τονικά συνεπές.
    """
    ta, ma = P.estimate_key(anchor["melody"], anchor["accomp"])
    to, mo = P.estimate_key(other["melody"], other["accomp"])
    k = (P.reference_pc(ta, ma) - P.reference_pc(to, mo)) % 12
    if k > 6:
        k -= 12
    return transpose(other, k)


def partial_copy(anchor, other, fraction, rng):
    """
    Αντιγράφει ένα συνεχές απόσπασμα της μελωδίας του anchor (ποσοστό
    fraction, στρογγυλοποιημένο σε ολόκληρα μέτρα) και το τοποθετεί σε
    ΤΥΧΑΙΑ θέση μέσα στη μελωδία ενός άλλου τραγουδιού. Η συνοδεία προέρχεται
    εξ ολοκλήρου από το άλλο τραγούδι. Προσομοιώνει την περίπτωση όπου
    αντιγράφεται μόνο ένα χαρακτηριστικό μοτίβο (hook) — πρβλ. την υπόθεση
    Pelham του ΔΕΕ (C-476/17) για τη χρήση σύντομων αποσπασμάτων.

    Επιστρέφει (νέο τμήμα, πραγματικό ποσοστό αντιγραφής).
    """
    other = align_to(other, anchor)
    n_bars = int(np.clip(round(fraction * T / BAR), 1, T // BAR))
    src = rng.randrange(0, T // BAR - n_bars + 1) * BAR   # θέση στο πρωτότυπο
    dst = rng.randrange(0, T // BAR - n_bars + 1) * BAR   # θέση στο νέο έργο
    length = n_bars * BAR
    a, o = anchor["melody"], other["melody"]
    copied = a[(a[:, 0] >= src) & (a[:, 0] < src + length)].astype(np.int32).copy()
    copied[:, 0] += dst - src
    kept = o[(o[:, 0] < dst) | (o[:, 0] >= dst + length)]
    new = copy_segment(other)
    new["melody"] = fix_notes(np.concatenate([copied, kept]), monophonic=True)
    return new, length / T


def swap_accompaniment(anchor, other):
    """Κρατά τη μελωδία του anchor και αντικαθιστά τη συνοδεία με άλλου τραγουδιού."""
    other = align_to(other, anchor)
    new = copy_segment(anchor)
    new["accomp"] = other["accomp"].astype(np.int32).copy()
    return new


def maybe_time_shift(seg, rng):
    """Με πιθανότητα TIME_SHIFT_PROB εφαρμόζει τυχαία χρονική μετατόπιση ±1..4 βημάτων."""
    if rng.random() >= config.TIME_SHIFT_PROB:
        return seg
    s = rng.choice([i for i in range(-config.MAX_TIME_SHIFT,
                                     config.MAX_TIME_SHIFT + 1) if i != 0])
    return time_shift(seg, s)


# ===========================================================================
# 2. Επιλογή αρνητικών παραδειγμάτων
# ===========================================================================
def harmony_profile(enc):
    """
    Αρμονικό «αποτύπωμα» τμήματος: μέσο chroma ανά μέτρο (8 × 12),
    κανονικοποιημένο ως διάνυσμα. Εντοπίζει τμήματα με παρόμοια αρμονική πορεία.
    """
    v = enc["harmony"].reshape(T // BAR, BAR, 12).mean(axis=1).flatten()
    return v / (np.linalg.norm(v) + 1e-8)


def melody_profile(enc):
    """
    Μελωδικό «αποτύπωμα» τμήματος: ιστόγραμμα διαστημάτων σταθμισμένο με τη
    διάρκεια, ιστόγραμμα τονικών βαθμίδων και ρυθμικό μοτίβο εναρξεων ανά
    μέτρο. Εντοπίζει τμήματα με παρόμοιο ύφος μελωδίας, χωρίς να αποτελούν
    αντίγραφα.
    """
    iv = np.bincount(enc["interval"], minlength=config.N_INTERVAL_CLASSES)[1:]
    deg = np.bincount(enc["melody"], minlength=config.N_PITCH_CLASSES)[1:]
    rhythm = enc["rhythm"][:, 0].reshape(T // BAR, BAR).mean(axis=0)
    parts = [iv / (iv.sum() + 1e-8), deg / (deg.sum() + 1e-8), rhythm]
    v = np.concatenate([x / (np.linalg.norm(x) + 1e-8) for x in parts])
    return v / (np.linalg.norm(v) + 1e-8)


def pick_other(pool, song_id, rng):
    """Επιλέγει τυχαίο τμήμα από ΔΙΑΦΟΡΕΤΙΚΟ τραγούδι."""
    while True:
        cand = rng.choice(pool)
        if cand["song_id"] != song_id:
            return cand


def most_similar_other(anchor, pool, profiles, rng):
    """
    Από HARD_NEG_POOL τυχαία τμήματα άλλων τραγουδιών επιλέγει εκείνο με τη
    μεγαλύτερη ομοιότητα συνημιτόνου (cosine similarity) στο δοσμένο προφίλ.
    """
    target = profiles[id(anchor)]
    best, best_sim = None, -np.inf
    for _ in range(config.HARD_NEG_POOL):
        cand = pick_other(pool, anchor["song_id"], rng)
        sim = float(target @ profiles[id(cand)])
        if sim > best_sim:
            best, best_sim = cand, sim
    return best


def light_noise(seg, rng):
    """Ελαφριά τυχαία αλλοίωση (για να μη διακρίνονται τα θετικά από ίχνη επεξεργασίας)."""
    r = rng.random()
    if r < 1 / 3:
        return melodic_variation(seg, rng.uniform(*config.POS_MELODIC_P), rng)
    if r < 2 / 3:
        return rhythmic_variation(seg, rng.uniform(*config.POS_RHYTHMIC_P), rng)
    return ornamentation(seg, rng.uniform(*config.POS_ORNAMENT_P),
                         rng.uniform(*config.POS_OCTAVE_P), rng)


# ===========================================================================
# 3. Κατασκευή ζευγών
# ===========================================================================
EXTRA_NEGATIVE_TYPES = ["remelodized"]   # έκδοση v2
POSITIVE_TYPES = ["transposed", "melodic", "rhythmic", "ornamented",
                  "combined", "partial", "new_accomp"]
NEGATIVE_TYPES = ["random", "hard_harmony", "hard_melody"]


def make_positive(anchor, pool, rng, kind):
    """Δημιουργεί το δεύτερο μέλος ενός θετικού ζεύγους. Επιστρέφει (τμήμα, βαθμός)."""
    if kind == "transposed":
        b, degree = anchor, 1.0
    elif kind == "melodic":
        p = rng.uniform(*config.POS_MELODIC_P)
        b, degree = melodic_variation(anchor, p, rng), 1 - p
    elif kind == "rhythmic":
        p = rng.uniform(*config.POS_RHYTHMIC_P)
        b, degree = rhythmic_variation(anchor, p, rng), 1 - p
    elif kind == "ornamented":
        p = rng.uniform(*config.POS_ORNAMENT_P)
        b = ornamentation(anchor, p, rng.uniform(*config.POS_OCTAVE_P), rng)
        degree = 1 - p / 2       # τα στολίδια αλλάζουν μισή διάρκεια νότας
    elif kind == "combined":
        p1, p2, p3 = (rng.uniform(0.05, 0.2) for _ in range(3))
        b = melodic_variation(anchor, p1, rng)
        b = rhythmic_variation(b, p2, rng)
        b = ornamentation(b, p3, 0.0, rng)
        degree = (1 - p1) * (1 - p2) * (1 - p3 / 2)
    elif kind == "partial":
        other = pick_other(pool, anchor["song_id"], rng)
        b, degree = partial_copy(anchor, other,
                                 rng.uniform(*config.POS_PARTIAL_FRACTION), rng)
    elif kind == "new_accomp":
        other = pick_other(pool, anchor["song_id"], rng)
        b, degree = swap_accompaniment(anchor, other), 1.0
    else:
        raise ValueError(kind)
    return random_transpose(maybe_time_shift(b, rng), rng), degree


def make_negative(anchor, pool, profiles, rng, kind):
    """Δημιουργεί το δεύτερο μέλος ενός αρνητικού ζεύγους."""
    if kind == "random":
        b = pick_other(pool, anchor["song_id"], rng)
    elif kind == "hard_harmony":
        b = most_similar_other(anchor, pool, profiles["harmony"], rng)
    elif kind == "hard_melody":
        b = most_similar_other(anchor, pool, profiles["melody"], rng)
    else:
        raise ValueError(kind)
    if rng.random() < 0.5:
        b = light_noise(b, rng)
    return random_transpose(maybe_time_shift(b, rng), rng), 0.0


def compute_profiles(segments):
    """Προϋπολογίζει τα αρμονικά και μελωδικά αποτυπώματα όλων των τμημάτων."""
    profiles = {"harmony": {}, "melody": {}}
    for s in segments:
        enc = P.encode_segment(s)
        profiles["harmony"][id(s)] = harmony_profile(enc)
        profiles["melody"][id(s)] = melody_profile(enc)
    return profiles


def build_pairs(segments, seed):
    """
    Για κάθε τμήμα αναφοράς (anchor) δημιουργεί ένα θετικό και ένα αρνητικό
    ζεύγος, ώστε οι κλάσεις να είναι ισορροπημένες. Οι τύποι εναλλάσσονται
    κυκλικά για ίσες αναλογίες.
    """
    rng = random.Random(seed)
    profiles = compute_profiles(segments)
    pairs = []
    for i, anchor in enumerate(segments):
        pos_kind = POSITIVE_TYPES[i % len(POSITIVE_TYPES)]
        neg_kind = NEGATIVE_TYPES[i % len(NEGATIVE_TYPES)]
        b, degree = make_positive(anchor, segments, rng, pos_kind)
        pairs.append({"a": anchor, "b": b, "label": 1, "type": pos_kind,
                      "degree": degree})
        b, degree = make_negative(anchor, segments, profiles, rng, neg_kind)
        pairs.append({"a": anchor, "b": b, "label": 0, "type": neg_kind,
                      "degree": degree})
    rng.shuffle(pairs)
    return pairs


def remelodize(anchor, pool, rng):
    """
    Αρνητικό v2: κρατά ρυθμό και θέσεις νοτών, αλλάζει 70–100% των τονικών
    υψών. Με πιθανότητα 50% η συνοδεία αντικαθίσταται από άλλο τραγούδι,
    ώστε το μοντέλο να μη στηρίζεται ούτε στην (ίδια) συνοδεία.
    """
    b = melodic_variation(anchor, rng.uniform(*config.NEG_REMELODIZE_P), rng)
    if rng.random() < 0.5:
        other = align_to(pick_other(pool, anchor["song_id"], rng), anchor)
        b["accomp"] = other["accomp"].astype(np.int32).copy()
    return b


def build_remelodized(segments, seed):
    """Επιπλέον αρνητικά ζεύγη v2 για ποσοστό EXTRA_FRACTION των τμημάτων."""
    rng = random.Random(seed)
    anchors = rng.sample(segments, int(len(segments) * config.EXTRA_FRACTION))
    pairs = []
    for anchor in anchors:
        b = random_transpose(maybe_time_shift(remelodize(anchor, segments, rng), rng), rng)
        pairs.append({"a": anchor, "b": b, "label": 0, "type": "remelodized",
                      "degree": 0.0})
    return pairs


def build_sweeps(segments, seed):
    """
    Σύνολα «σάρωσης» (sweep) για την ανάλυση ευαισθησίας στο test set:
      - melodic : μελωδική αλλοίωση με p = 0, 0.1, ..., 1.0
      - partial : αντιγραφή 0, 1, 2, ..., 8 μέτρων μέσα σε άλλο τραγούδι
    Δείχνουν πώς μεταβάλλεται η βαθμολογία ομοιότητας καθώς μειώνεται το
    ποσοστό αντιγραμμένου υλικού (label = -1: χωρίς δυαδική ετικέτα).
    """
    rng = random.Random(seed)
    anchors = rng.sample(segments, min(config.SWEEP_ANCHORS, len(segments)))
    sweeps = {"melodic": [], "partial": []}
    for anchor in anchors:
        for p in config.SWEEP_LEVELS:
            b = random_transpose(melodic_variation(anchor, p, rng), rng)
            sweeps["melodic"].append({"a": anchor, "b": b, "label": -1,
                                      "type": "sweep_melodic", "degree": 1 - p})
        other = pick_other(segments, anchor["song_id"], rng)
        for n_bars in range(0, T // BAR + 1):
            if n_bars == 0:
                b, frac = align_to(other, anchor), 0.0
            else:
                b, frac = partial_copy(anchor, other, n_bars * BAR / T, rng)
            sweeps["partial"].append({"a": anchor, "b": random_transpose(b, rng),
                                      "label": -1, "type": "sweep_partial",
                                      "degree": frac})
    return sweeps


# ===========================================================================
# 4. Κωδικοποίηση και αποθήκευση
# ===========================================================================
def encode_pairs(pairs):
    """
    Κωδικοποιεί τα ζεύγη σε πίνακες numpy για το νευρωνικό δίκτυο.
    Για εξοικονόμηση χώρου, ρυθμός και αρμονία αποθηκεύονται ως uint8 (×255).
    """
    out = defaultdict(list)
    for pr in pairs:
        for side in ("a", "b"):
            e = P.encode_segment(pr[side])
            out[f"{side}_melody"].append(e["melody"].astype(np.int8))
            out[f"{side}_interval"].append(e["interval"].astype(np.int8))
            out[f"{side}_rhythm"].append(np.round(e["rhythm"] * 255).astype(np.uint8))
            out[f"{side}_harmony"].append(np.round(e["harmony"] * 255).astype(np.uint8))
        out["label"].append(pr["label"])
        out["degree"].append(pr["degree"])
        out["type"].append(pr["type"])
        out["song_a"].append(pr["a"]["song_id"])
        out["song_b"].append(pr["b"]["song_id"])
    return {k: np.array(v) for k, v in out.items()}


def save_pairs(pairs, name):
    """Αποθηκεύει τα κωδικοποιημένα ζεύγη (.npz) και τις νότες (.pkl, για το ηχητικό baseline)."""
    np.savez_compressed(config.PAIRS_DIR / f"{name}.npz", **encode_pairs(pairs))
    with open(config.PAIRS_DIR / f"{name}_notes.pkl", "wb") as f:
        pickle.dump(pairs, f)


def load_pairs(name):
    """
    Φορτώνει ζεύγη για το μοντέλο. Επιστρέφει λεξικό με:
        a_melody, b_melody     : (N, T) int   (δείκτες κλάσεων ύψους)
        a_interval, b_interval : (N, T) int   (δείκτες μελωδικών διαστημάτων)
        a_rhythm, b_rhythm     : (N, T, 3) float32 στο [0, 1]
        a_harmony, b_harmony   : (N, T, 12) float32 στο [0, 1]
        label, degree, type, song_a, song_b
    """
    data = dict(np.load(config.PAIRS_DIR / f"{name}.npz"))
    for side in ("a", "b"):
        for key in ("melody", "interval"):
            data[f"{side}_{key}"] = data[f"{side}_{key}"].astype(np.int32)
        for key in ("rhythm", "harmony"):
            data[f"{side}_{key}"] = data[f"{side}_{key}"].astype(np.float32) / 255.0
    return data


def load_pair_notes(name):
    """Φορτώνει τα ζεύγη σε μορφή νοτών (για σύνθεση ήχου / οπτικοποίηση)."""
    with open(config.PAIRS_DIR / f"{name}_notes.pkl", "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Δημιουργία ζευγών")
    parser.add_argument("--extra", action="store_true",
                        help="μόνο τα επιπλέον αρνητικά της έκδοσης v2")
    args = parser.parse_args()
    config.ensure_dirs()
    segments, splits = P.load_segments()
    by_split = {name: [s for s in segments if s["song_id"] in set(ids)]
                for name, ids in splits.items()}

    # Επιπλέον αρνητικά v2 (ξεχωριστά αρχεία, ώστε τα αρχικά ζεύγη να μένουν
    # ίδια και τα αποτελέσματα της v1 να παραμένουν συγκρίσιμα).
    for i, name in enumerate(("train", "val", "test")):
        extra = build_remelodized(by_split[name], seed=config.SEED + 20 + i)
        save_pairs(extra, f"{name}_remelodized")
        print(f"{name}_remelodized: {len(extra)} αρνητικά ζεύγη")
    if args.extra:
        return

    for i, name in enumerate(("train", "val", "test")):
        pairs = build_pairs(by_split[name], seed=config.SEED + i)
        save_pairs(pairs, name)
        counts = defaultdict(int)
        for pr in pairs:
            counts[pr["type"]] += 1
        print(f"{name:5s}: {len(pairs):6d} ζεύγη  {dict(counts)}")

    sweeps = build_sweeps(by_split["test"], seed=config.SEED + 10)
    for kind, pairs in sweeps.items():
        save_pairs(pairs, f"sweep_{kind}")
        print(f"sweep_{kind}: {len(pairs)} ζεύγη")
    print(f"Αποθηκεύτηκαν στο {config.PAIRS_DIR}")


if __name__ == "__main__":
    main()
