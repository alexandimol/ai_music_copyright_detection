# -*- coding: utf-8 -*-
"""
plot_style.py
=============
Κοινή εμφάνιση για όλα τα γραφήματα της εργασίας (matplotlib), ώστε οι
εικόνες να είναι ομοιόμορφες και ευανάγνωστες και σε ασπρόμαυρη εκτύπωση
(κάθε σειρά έχει και διαφορετικό σύμβολο/γραμμή, όχι μόνο χρώμα).
"""

import matplotlib

matplotlib.use("Agg")   # αποθήκευση σε αρχείο χωρίς παράθυρο
import matplotlib.pyplot as plt  # noqa: E402

# Κατηγορική παλέτα με σταθερή σειρά (ελεγμένη για αχρωματοψία).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

# Σταθερή αντιστοίχιση μεθόδου → χρώμα (το χρώμα ακολουθεί τη μέθοδο).
METHOD_ORDER = ["siamese_v2", "siamese_melody_v2", "siamese", "siamese_melody",
                "smith_waterman", "interval_match", "audio_chroma_dtw",
                "harmony_cosine"]
METHOD_LABELS = {
    "siamese_v2": "Siamese v2 (μελωδία+ρυθμός+αρμονία)",
    "siamese_melody_v2": "Siamese v2 (μόνο μελωδία)",
    "siamese": "Siamese v1 (μελωδία+ρυθμός+αρμονία)",
    "siamese_melody": "Siamese v1 (μόνο μελωδία)",
    "smith_waterman": "Smith–Waterman (διαστήματα)",
    "interval_match": "Αντιστοίχιση διαστημάτων",
    "audio_chroma_dtw": "Ήχος: chroma + DTW",
    "harmony_cosine": "Μόνο αρμονία (chroma)",
}
TYPE_LABELS = {
    "transposed": "μεταφορά",
    "melodic": "μελωδική αλλοίωση",
    "rhythmic": "ρυθμική αλλοίωση",
    "ornamented": "στολίδια",
    "combined": "συνδυασμός",
    "partial": "μερική αντιγραφή",
    "new_accomp": "νέα συνοδεία",
    "random": "τυχαίο",
    "hard_harmony": "ίδια αρμονία",
    "hard_melody": "όμοιο μελωδικό ύφος",
    "remelodized": "νέα μελωδία, ίδιος ρυθμός",
}
BRANCH_COLORS = {"melody": COLORS[0], "rhythm": COLORS[1], "harmony": COLORS[2]}
BRANCH_LABELS = {"melody": "Μελωδία", "rhythm": "Ρυθμός", "harmony": "Αρμονία"}


def method_style(name):
    """Χρώμα, σύμβολο και τύπος γραμμής για μια μέθοδο."""
    i = METHOD_ORDER.index(name) if name in METHOD_ORDER else len(METHOD_ORDER)
    i %= len(COLORS)
    return {"color": COLORS[i], "marker": MARKERS[i], "linestyle": LINESTYLES[i]}


def setup():
    """Εφαρμόζει τις κοινές ρυθμίσεις εμφάνισης."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
    })


setup()
