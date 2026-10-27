# Ανίχνευση πιθανής παραβίασης πνευματικών δικαιωμάτων σε μουσική που παράγεται από AI

Κώδικας της διπλωματικής εργασίας «AI-Generated Music: Τεχνολογία, Επιπτώσεις
και Ανίχνευση Παραβίασης Πνευματικών Δικαιωμάτων» (MSc Artificial Intelligence).

## 1. Εγκατάσταση

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## 2. Δεδομένα

Χρησιμοποιείται το σύνολο δεδομένων **POP909** (Wang et al., 2020): 909 ποπ
τραγούδια σε μορφή MIDI με ξεχωριστό κομμάτι μελωδίας.

```bash
git clone https://github.com/music-x-lab/POP909-Dataset.git data/POP909-Dataset
```

## Γρήγορη επαλήθευση των αποτελεσμάτων

Το αποθετήριο περιλαμβάνει τα εκπαιδευμένα μοντέλα (`outputs/models/`), τις
βαθμολογίες όλων των μεθόδων και τους πίνακες (`outputs/results/`), τα σχήματα
(`outputs/figures/`) και τις μελωδίες των generators (`outputs/generated/`).
Το POP909 και τα ζεύγη που προκύπτουν από αυτό δεν περιλαμβάνονται, ώστε να μην
αναδιανέμονται έργα του συνόλου δεδομένων. Τα ζεύγη δημιουργούνται ξανά
πανομοιότυπα, χάρη στο σταθερό seed.

Για να αναπαραχθούν οι πίνακες και τα σχήματα της εργασίας χωρίς νέα εκπαίδευση
(περίπου 10–15 λεπτά σε CPU):

```bash
git clone https://github.com/music-x-lab/POP909-Dataset.git data/POP909-Dataset
python preprocessing.py      # τμήματα 8 μέτρων
python pairs.py              # όλα τα ζεύγη (και τα αρνητικά «νέα μελωδία» της v2)
python evaluate.py           # Πίνακες 7.1–7.3, Σχήματα 7.1–7.3
python robustness.py         # Πίνακες 7.5–7.6 (επαναλήψεις με seed, bootstrap)
python xai.py                # Σχήματα 7.4–7.5 (επεξηγησιμότητα)
python memorization.py       # Πίνακας 7.4, Σχήματα 7.6–7.7
```

Η πλήρης διαδικασία, με εκπαίδευση όλων των μοντέλων από την αρχή, περιγράφεται
στην επόμενη ενότητα.

## 3. Εκτέλεση (με τη σειρά)

| Βήμα | Εντολή | Τι κάνει | Χρόνος (CPU) |
|---|---|---|---|
| 1 | `python preprocessing.py` | MIDI → κβαντισμένα τμήματα 8 μέτρων, διαχωρισμός train/val/test ανά τραγούδι | ~1 λεπτό |
| 2 | `python pairs.py` | Δημιουργία επισημασμένων ζευγών με ελεγχόμενους μετασχηματισμούς | ~3 λεπτά |
| 3 | `python baselines.py` | Κλασικές μέθοδοι: αντιστοίχιση διαστημάτων, Smith–Waterman, αρμονία, ήχος (chroma + DTW) | ~15–20 λεπτά |
| 4 | `python train.py --quick` | Γρήγορη δοκιμή ότι το TensorFlow και το μοντέλο λειτουργούν | ~1–3 λεπτά |
| 5 | `python train.py` | Πλήρης εκπαίδευση του Siamese ανιχνευτή | ~15–40 λεπτά |
| 5β | `python train.py --branches melody --name siamese_melody` | Πείραμα αφαίρεσης (μόνο μελωδία) | ~2 λεπτά |
| 6 | `python xai.py` | Εικόνες επεξήγησης (πίνακες ομοιότητας, Integrated Gradients, συνεισφορές) | ~1 λεπτό |
| 7 | `python generator.py --name gen_full` | Generator LSTM σε όλα τα δεδομένα εκπαίδευσης + 200 μελωδίες | ~15–25 λεπτά |
| 7β | `python generator.py --name gen_small --subset 100 --epochs 400 --batch_size 16 --dropout 0` | Generator με υπερπροσαρμογή (θετικός έλεγχος απομνημόνευσης) | ~3–8 λεπτά |
| 8 | `python memorization.py` | Αντιγράφουν οι generators τα δεδομένα τους; (+ ομάδα ελέγχου) | ~2–5 λεπτά |
| 9 | `python evaluate.py` | Τελικοί πίνακες και γραφήματα όλων των μεθόδων | <1 λεπτό |

### Έκδοση v2 (μετά την ανάλυση της v1)

Οι καμπύλες sweep έδειξαν ότι η v1 θεωρούσε «αντιγραφή» και τμήματα με τον ίδιο
ρυθμό αλλά εντελώς νέα μελωδία. Η v2 προσθέτει τέτοια αρνητικά ζεύγη:

| Βήμα | Εντολή |
|---|---|
| v2-1 | `python pairs.py --extra` |
| v2-2 | `python baselines.py --splits test_remelodized` |
| v2-3 | `python train.py --name siamese --eval_only` και `python train.py --name siamese_melody --eval_only` |
| v2-4 | `python train.py --name siamese_v2 --extra remelodized` |
| v2-5 | `python train.py --branches melody --name siamese_melody_v2 --extra remelodized` |
| v2-6 | `python xai.py`, `python memorization.py`, `python evaluate.py` |

### Εκτίμηση αβεβαιότητας (seed και bootstrap)

Κάθε Siamese μοντέλο εκπαιδεύεται ξανά με 3 επιπλέον τιμές seed (1, 2, 3). Τα ζεύγη
μένουν ίδια· αλλάζει μόνο η αρχικοποίηση και η σειρά εκπαίδευσης. Οι επαναλήψεις
αποθηκεύονται με επίθημα `_s1`, `_s2`, `_s3` και ΔΕΝ αλλάζουν τους πίνακες του
`evaluate.py`. Στο PowerShell (από τον φάκελο του κώδικα):

```
foreach ($s in 1,2,3) {
    python train.py --name siamese_v2 --extra remelodized --seed $s
    python train.py --branches melody --name siamese_melody_v2 --extra remelodized --seed $s
    python train.py --name siamese --seed $s
    python train.py --branches melody --name siamese_melody --seed $s
}
python robustness.py
```

Το `robustness.py` παράγει τα `table_seeds.csv` (μέσος όρος ± τυπική απόκλιση),
`table_seeds_runs.csv` (ανά εκτέλεση), `table_seeds_contrib.csv` (συνεισφορές
κλάδων) και `table_bootstrap.csv` (διαστήματα εμπιστοσύνης 95%, bootstrap σε
επίπεδο τραγουδιού, για όλες τις μεθόδους).

Όλα τα αποτελέσματα αποθηκεύονται στο `outputs/results/` (πίνακες .csv) και στο
`outputs/figures/` (γραφήματα .png).

## 4. Δομή

```
config.py          όλες οι παράμετροι σε ένα σημείο
preprocessing.py   ανάγνωση MIDI, κβαντισμός, τμηματοποίηση, εκτίμηση τονικότητας, κωδικοποίηση
pairs.py           μετασχηματισμοί (μεταφορά, μελωδική/ρυθμική αλλοίωση, στολίδια,
                   μερική αντιγραφή, αλλαγή συνοδείας) και «δύσκολα» αρνητικά ζεύγη
baselines.py       κλασικές μέθοδοι ομοιότητας (σημεία αναφοράς)
model.py           Siamese δίκτυο με πίνακες ομοιότητας ανά διάσταση
train.py           εκπαίδευση, αξιολόγηση, αποθήκευση βαθμολογιών
xai.py             επεξηγήσεις: πίνακες ομοιότητας, Integrated Gradients, συνεισφορές
generator.py       generator μελωδιών LSTM (AI-generated μουσική για το πείραμα)
memorization.py    πείραμα απομνημόνευσης: ομοιότητα παραγόμενων μελωδιών με τα δεδομένα εκπαίδευσης
evaluate.py        συγκριτικοί πίνακες και γραφήματα
robustness.py      αβεβαιότητα: επαναλήψεις με διαφορετικό seed και bootstrap διαστήματα εμπιστοσύνης
plot_style.py      κοινή εμφάνιση γραφημάτων
outputs/           παραγόμενα αρχεία (δημιουργείται αυτόματα)
```

## 5. Αναπαράσταση

Κάθε τμήμα (8 μέτρα = 128 δέκατα έκτα) κωδικοποιείται σε τέσσερα κανάλια,
που αντιστοιχούν στα στοιχεία μουσικής ομοιότητας που εξετάζονται και σε
νομικό επίπεδο:

| Κανάλι | Διάσταση | Περιγραφή |
|---|---|---|
| melody | 128 | τονικό ύψος σχετικά με την εκτιμώμενη τονική (αναλλοίωτο στη μεταφορά) |
| interval | 128 | μελωδικό διάστημα από την προηγούμενη νότα |
| rhythm | 128 × 3 | ενάρξεις και διάρκειες μελωδίας, ρυθμική πυκνότητα συνοδείας |
| harmony | 128 × 12 | chroma της συνοδείας ως προς την τονική |
