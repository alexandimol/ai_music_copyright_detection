# -*- coding: utf-8 -*-
"""
preprocessing.py
================
Προεπεξεργασία του συνόλου δεδομένων POP909 (Wang et al., 2020).

Ροή επεξεργασίας για κάθε τραγούδι:
    1. Ανάγνωση του αρχείου MIDI με τη βιβλιοθήκη pretty_midi (Raffel & Ellis, 2014).
    2. Κβαντισμός (quantization) των νοτών σε πλέγμα δεκάτων έκτων, με βάση
       τους χρόνους (beats) του κομματιού.
    3. Διαχωρισμός σε τμήματα (segments) των 8 μέτρων με 50% επικάλυψη.
    4. Αποθήκευση των τμημάτων σε μορφή λίστας νοτών (onset, duration, pitch),
       ώστε οι μετασχηματισμοί του pairs.py να εφαρμόζονται σε επίπεδο νότας.

Επιπλέον, το αρχείο περιέχει τη συνάρτηση encode_segment(), η οποία μετατρέπει
ένα τμήμα στην αναπαράσταση εισόδου του νευρωνικού δικτύου, με τρία ξεχωριστά
«κανάλια» που αντιστοιχούν στα στοιχεία που εξετάζει η νομολογία για τη
μουσική ομοιότητα: μελωδία, ρυθμός και αρμονία.

Εκτέλεση:
    python preprocessing.py
"""

import pickle
import random
from pathlib import Path

import numpy as np
import pretty_midi

import config

# ---------------------------------------------------------------------------
# Προφίλ τονικότητας Krumhansl–Kessler (Krumhansl, 1990).
# Εκφράζουν πόσο «σταθερή» ακούγεται κάθε βαθμίδα της κλίμακας σε ματζόρε
# και μινόρε τονικότητα. Χρησιμοποιούνται για την εκτίμηση της τονικότητας
# (αλγόριθμος Krumhansl–Schmuckler).
# ---------------------------------------------------------------------------
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _zscore(v):
    return (v - v.mean()) / v.std()


# Πίνακας 24 × 12 με όλα τα περιστρεμμένα και τυποποιημένα προφίλ:
# γραμμές 0–11 = ματζόρε με τονική C..B, γραμμές 12–23 = μινόρε.
_KEY_PROFILES_Z = np.array(
    [_zscore(np.roll(KK_MAJOR, t)) for t in range(12)] +
    [_zscore(np.roll(KK_MINOR, t)) for t in range(12)])

# Διαστήματα (σε ημιτόνια από την τονική) της ματζόρε και της φυσικής μινόρε
# κλίμακας. Χρησιμοποιούνται από τις μελωδικές παραλλαγές του pairs.py.
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


# ===========================================================================
# 1. Κβαντισμός τραγουδιού
# ===========================================================================
def build_step_grid(pm):
    """
    Δημιουργεί το χρονικό πλέγμα των δεκάτων έκτων για ένα τραγούδι.

    Κάθε διάστημα μεταξύ δύο διαδοχικών χρόνων (beats) χωρίζεται σε
    STEPS_PER_BEAT ίσα μέρη. Έτσι το πλέγμα ακολουθεί τυχόν αλλαγές
    ρυθμού (tempo) μέσα στο κομμάτι.

    Επιστρέφει πίνακα με τον χρόνο (σε δευτερόλεπτα) κάθε βήματος.
    """
    beats = pm.get_beats()
    if len(beats) < 2:
        raise ValueError("Το αρχείο δεν περιέχει αρκετούς χρόνους (beats).")
    # Προσθέτουμε έναν επιπλέον χρόνο στο τέλος (παρέκταση), ώστε να
    # καλύπτεται και το τελευταίο διάστημα.
    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    grid = []
    for b0, b1 in zip(beats[:-1], beats[1:]):
        grid.extend(np.linspace(b0, b1, config.STEPS_PER_BEAT, endpoint=False))
    grid.append(beats[-1])
    return np.asarray(grid)


def time_to_step(t, grid):
    """Αντιστοιχίζει χρόνο (δευτερόλεπτα) στο πλησιέστερο βήμα του πλέγματος."""
    return int(round(float(np.interp(t, grid, np.arange(len(grid))))))


def quantize_notes(notes, grid):
    """
    Κβαντίζει μια λίστα νοτών pretty_midi.

    Επιστρέφει πίνακα (n, 3) με στήλες [onset_step, duration_steps, pitch].
    Η ελάχιστη διάρκεια είναι ένα βήμα.
    """
    out = []
    for n in notes:
        on = time_to_step(n.start, grid)
        off = time_to_step(n.end, grid)
        out.append((on, max(1, off - on), n.pitch))
    if not out:
        return np.zeros((0, 3), dtype=np.int32)
    arr = np.array(out, dtype=np.int32)
    return arr[np.lexsort((arr[:, 2], arr[:, 0]))]  # ταξινόμηση κατά onset


def make_monophonic(melody):
    """
    Εξασφαλίζει ότι η μελωδία είναι μονοφωνική (μία νότα κάθε στιγμή).

    - Αν δύο νότες ξεκινούν στο ίδιο βήμα, κρατάμε την υψηλότερη
      (κανόνας «skyline», συνήθης στη βιβλιογραφία ανάκτησης μουσικής).
    - Αν μια νότα διαρκεί πέρα από την αρχή της επόμενης, την περικόπτουμε.
    """
    if len(melody) == 0:
        return melody
    by_onset = {}
    for on, dur, p in melody:
        if on not in by_onset or p > by_onset[on][2]:
            by_onset[on] = [on, dur, p]
    mono = np.array([by_onset[k] for k in sorted(by_onset)], dtype=np.int32)
    for i in range(len(mono) - 1):
        gap = mono[i + 1, 0] - mono[i, 0]
        mono[i, 1] = max(1, min(mono[i, 1], gap))
    return mono


def load_song(midi_path):
    """
    Φορτώνει και κβαντίζει ένα τραγούδι του POP909.

    Επιστρέφει λεξικό με:
        melody       : (n, 3) νότες της κύριας μελωδίας (κομμάτι MELODY)
        accomp       : (m, 3) νότες της συνοδείας (κομμάτια BRIDGE + PIANO)
        grid         : χρόνοι των βημάτων σε δευτερόλεπτα
        n_steps      : συνολικό πλήθος βημάτων
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    grid = build_step_grid(pm)

    melody_notes, accomp_notes = [], []
    for inst in pm.instruments:
        name = inst.name.strip().upper()
        if name == config.MELODY_TRACK:
            melody_notes.extend(inst.notes)
        elif name in config.ACCOMP_TRACKS:
            accomp_notes.extend(inst.notes)

    melody = make_monophonic(quantize_notes(melody_notes, grid))
    accomp = quantize_notes(accomp_notes, grid)
    return {"melody": melody, "accomp": accomp, "grid": grid,
            "n_steps": len(grid) - 1}


# ===========================================================================
# 2. Διαχωρισμός σε τμήματα
# ===========================================================================
def slice_notes(notes, s0, length, keep_partial):
    """
    Επιλέγει τις νότες που ανήκουν στο παράθυρο [s0, s0 + length) και
    μετατοπίζει τους χρόνους ώστε το τμήμα να ξεκινά από το βήμα 0.

    keep_partial=True : κρατά και νότες που ξεκίνησαν πριν από το παράθυρο
                        αλλά ηχούν μέσα σε αυτό (χρήσιμο για τη συνοδεία/αρμονία).
    keep_partial=False: κρατά μόνο νότες που ξεκινούν μέσα στο παράθυρο
                        (για τη μελωδία, ώστε να μη «κόβονται» νότες).
    """
    if len(notes) == 0:
        return notes.copy()
    on, dur = notes[:, 0], notes[:, 1]
    if keep_partial:
        mask = (on < s0 + length) & (on + dur > s0)
    else:
        mask = (on >= s0) & (on < s0 + length)
    sel = notes[mask].copy()
    start = np.maximum(sel[:, 0], s0)
    end = np.minimum(sel[:, 0] + sel[:, 1], s0 + length)
    sel[:, 0] = start - s0
    sel[:, 1] = np.maximum(1, end - start)
    return sel


def segment_song(song, song_id):
    """
    Χωρίζει ένα κβαντισμένο τραγούδι σε τμήματα σταθερού μήκους.

    Κάθε τμήμα είναι λεξικό με τα πεδία:
        song_id, seg_idx, start_step, melody, accomp, sec_per_step
    Το sec_per_step (μέση διάρκεια βήματος) χρειάζεται μόνο για τη σύνθεση
    ήχου στο ηχητικό baseline.
    """
    T = config.SEGMENT_STEPS
    hop = config.HOP_BEATS * config.STEPS_PER_BEAT
    segments = []
    for seg_idx, s0 in enumerate(range(0, song["n_steps"] - T + 1, hop)):
        mel = slice_notes(song["melody"], s0, T, keep_partial=False)
        if len(mel) < config.MIN_MELODY_NOTES:
            continue  # τμήμα χωρίς ουσιαστική μελωδία (π.χ. εισαγωγή)
        acc = slice_notes(song["accomp"], s0, T, keep_partial=True)
        sec_per_step = (song["grid"][s0 + T] - song["grid"][s0]) / T
        segments.append({
            "song_id": song_id,
            "seg_idx": seg_idx,
            "start_step": s0,
            "melody": mel.astype(np.int16),
            "accomp": acc.astype(np.int16),
            "sec_per_step": float(sec_per_step),
        })
    return segments


# ===========================================================================
# 3. Εκτίμηση τονικότητας (Krumhansl–Schmuckler)
# ===========================================================================
def pitch_class_histogram(melody, accomp):
    """Ιστόγραμμα των 12 τονικών κλάσεων, σταθμισμένο με τη διάρκεια."""
    hist = np.zeros(12)
    for notes in (melody, accomp):
        for _, dur, p in notes:
            hist[p % 12] += dur
    return hist


def estimate_key(melody, accomp):
    """
    Εκτιμά την τονικότητα ενός τμήματος με τον αλγόριθμο Krumhansl–Schmuckler:
    υπολογίζεται η συσχέτιση Pearson του ιστογράμματος τονικών κλάσεων με
    τα 24 περιστρεμμένα προφίλ (12 ματζόρε + 12 μινόρε) και επιλέγεται το
    μέγιστο.

    Επιστρέφει (tonic_pc, mode) όπου tonic_pc ∈ {0..11} και mode ∈ {"major", "minor"}.
    """
    hist = pitch_class_histogram(melody, accomp)
    if hist.std() == 0:
        return 0, "major"
    # Συσχέτιση Pearson = μέσος όρος του γινομένου των τυποποιημένων (z-scores)
    # διανυσμάτων. Υπολογίζεται ταυτόχρονα για τα 24 προφίλ.
    z = (hist - hist.mean()) / hist.std()
    r = _KEY_PROFILES_Z @ z / 12.0
    best = int(np.argmax(r))
    return best % 12, ("major" if best < 12 else "minor")


def reference_pc(tonic, mode):
    """
    Τονική κλάση αναφοράς για την κωδικοποίηση.

    Οι σχετικές τονικότητες (π.χ. Ντο ματζόρε και Λα μινόρε) έχουν τον ίδιο
    οπλισμό και συχνά συγχέονται από τον αλγόριθμο. Για να μην αλλάζει η
    κωδικοποίηση όταν ο αλγόριθμος «πηδά» μεταξύ τους, χρησιμοποιούμε πάντα
    την τονική της αντίστοιχης ματζόρε (για μινόρε: τονική + 3 ημιτόνια).
    """
    return tonic if mode == "major" else (tonic + 3) % 12


def scale_of(tonic, mode):
    """Επιστρέφει το σύνολο των τονικών κλάσεων της κλίμακας."""
    base = MAJOR_SCALE if mode == "major" else MINOR_SCALE
    return {(tonic + d) % 12 for d in base}


# ===========================================================================
# 4. Κωδικοποίηση τμήματος για το νευρωνικό δίκτυο
# ===========================================================================
def encode_segment(seg):
    """
    Μετατρέπει ένα τμήμα στην αναπαράσταση εισόδου του μοντέλου.

    Η αναπαράσταση είναι ΑΝΑΛΛΟΙΩΤΗ ως προς τη μεταφορά τονικότητας
    (transposition invariant): τα τονικά ύψη εκφράζονται σχετικά με την
    εκτιμώμενη τονική. Η επιλογή αυτή είναι σκόπιμη, καθώς μια μεταφορά
    σε άλλη τονικότητα δεν αναιρεί την αντιγραφή μιας μελωδίας.

    Επιστρέφει λεξικό με:
        melody  : (T,)   int  — δείκτης κλάσης ύψους ανά βήμα (0 = παύση)
        interval: (T,)   int  — δείκτης μελωδικού διαστήματος από την
                                προηγούμενη νότα (0 = παύση/πρώτη νότα)
        rhythm: (T, 3) float — [έναρξη νότας μελωδίας, νότα σε διάρκεια,
                                  πυκνότητα εναρξεων συνοδείας]
        harmony : (T, 12) float — χρωματικό διάνυσμα (chroma) συνοδείας,
                                  περιστραμμένο ώστε η τονική αναφοράς
                                  (βλ. reference_pc) να είναι στη θέση 0
        tonic   : int   — εκτιμώμενη τονική (0 = C, 1 = C#, ...)
        mode    : str   — "major" ή "minor"
    """
    T = config.SEGMENT_STEPS
    mel, acc = seg["melody"], seg["accomp"]
    tonic, mode = estimate_key(mel, acc)
    ref_pc = reference_pc(tonic, mode)

    # --- Μελωδία -----------------------------------------------------------
    # Σημείο αναφοράς: η τονική κλάση αναφοράς στην οκτάβα που βρίσκεται
    # πλησιέστερα στη διάμεσο (median) της μελωδίας. Έτσι η μεταφορά κατά
    # k ημιτόνια αφήνει την κωδικοποίηση αμετάβλητη.
    melody = np.zeros(T, dtype=np.int32)
    interval = np.zeros(T, dtype=np.int32)
    rhythm = np.zeros((T, 3), dtype=np.float32)
    if len(mel):
        median = float(np.median(mel[:, 2]))
        ref = ref_pc + 12 * round((median - ref_pc) / 12)
        prev = None
        for on, dur, p in mel:
            rel = int(np.clip(p - ref, -config.PITCH_RANGE, config.PITCH_RANGE))
            idx = rel + config.PITCH_RANGE + 1          # 1..49
            end = min(T, on + dur)
            melody[on:end] = idx
            # Διάστημα από την προηγούμενη νότα (0 για την πρώτη νότα).
            if prev is not None:
                step = int(np.clip(p - prev, -config.INTERVAL_RANGE,
                                   config.INTERVAL_RANGE))
                interval[on:end] = step + config.INTERVAL_RANGE + 1  # 1..25
            prev = p
            rhythm[on, 0] = 1.0                          # έναρξη νότας
            rhythm[on:end, 1] = 1.0                      # νότα σε διάρκεια

    # --- Συνοδεία: ρυθμική πυκνότητα και αρμονία -----------------------------
    harmony = np.zeros((T, 12), dtype=np.float32)
    onsets = np.zeros(T, dtype=np.float32)
    for on, dur, p in acc:
        if 0 <= on < T:
            onsets[on] += 1
        harmony[on:min(T, on + dur), (p - ref_pc) % 12] += 1.0
    rhythm[:, 2] = np.minimum(onsets / 4.0, 1.0)   # κορεσμός στις 4 εναρξεις
    peak = harmony.max(axis=1, keepdims=True)
    harmony = np.divide(harmony, peak, out=np.zeros_like(harmony), where=peak > 0)

    return {"melody": melody, "interval": interval, "rhythm": rhythm,
            "harmony": harmony, "tonic": tonic, "mode": mode}


# ===========================================================================
# 5. Διαχωρισμός σε train / val / test σε επίπεδο τραγουδιού
# ===========================================================================
def split_songs(song_ids, seed=config.SEED):
    """Χωρίζει τυχαία (με σταθερό seed) τα τραγούδια σε τρία υποσύνολα."""
    ids = sorted(song_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(config.SPLIT_RATIOS["train"] * n))
    n_val = int(round(config.SPLIT_RATIOS["val"] * n))
    return {"train": sorted(ids[:n_train]),
            "val": sorted(ids[n_train:n_train + n_val]),
            "test": sorted(ids[n_train + n_val:])}


# ===========================================================================
# 6. Κύρια διαδικασία
# ===========================================================================
def build_dataset(pop909_dir=config.POP909_DIR, limit=None):
    """
    Επεξεργάζεται όλα τα τραγούδια του POP909 και επιστρέφει
    (λίστα τμημάτων, λεξικό διαχωρισμού).

    limit: προαιρετικό όριο πλήθους τραγουδιών (για γρήγορη δοκιμή).
    """
    if not Path(pop909_dir).exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε ο φάκελος του POP909: {pop909_dir}\n"
            "Κλωνοποιήστε το σύνολο δεδομένων μέσα στον φάκελο του έργου:\n"
            "  git clone https://github.com/music-x-lab/POP909-Dataset.git "
            "data/POP909-Dataset")
    song_dirs = sorted(p for p in Path(pop909_dir).iterdir() if p.is_dir())
    if limit:
        song_dirs = song_dirs[:limit]
    if not song_dirs:
        raise FileNotFoundError(f"Δεν βρέθηκαν τραγούδια στο {pop909_dir}")

    segments, failed = [], []
    for i, d in enumerate(song_dirs, 1):
        try:
            song = load_song(d / f"{d.name}.mid")
            segments.extend(segment_song(song, d.name))
        except Exception as exc:  # π.χ. κατεστραμμένο αρχείο
            failed.append((d.name, str(exc)))
        if i % 100 == 0 or i == len(song_dirs):
            print(f"  Επεξεργάστηκαν {i}/{len(song_dirs)} τραγούδια "
                  f"— {len(segments)} τμήματα")
    if failed:
        print(f"  Προσοχή: {len(failed)} τραγούδια απέτυχαν: {failed[:5]}")

    splits = split_songs({s["song_id"] for s in segments})
    return segments, splits


def load_segments(path=config.SEGMENTS_FILE):
    """Φορτώνει τα αποθηκευμένα τμήματα και τον διαχωρισμό."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το {path}.\n"
            "Τρέξτε πρώτα:  python preprocessing.py")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["segments"], data["splits"]


def main():
    config.ensure_dirs()
    print("Προεπεξεργασία POP909...")
    segments, splits = build_dataset()
    with open(config.SEGMENTS_FILE, "wb") as f:
        pickle.dump({"segments": segments, "splits": splits}, f)

    # Σύντομα στατιστικά για την ενότητα «Δεδομένα» της εργασίας.
    for name, ids in splits.items():
        ids = set(ids)
        n_seg = sum(s["song_id"] in ids for s in segments)
        print(f"  {name:5s}: {len(ids):4d} τραγούδια, {n_seg:6d} τμήματα")
    modes = [encode_segment(s)["mode"] for s in segments[:2000]]
    print(f"  Ποσοστό ματζόρε (πρώτα 2000 τμήματα): "
          f"{modes.count('major') / len(modes):.1%}")
    print(f"Αποθηκεύτηκε: {config.SEGMENTS_FILE}")


if __name__ == "__main__":
    main()
