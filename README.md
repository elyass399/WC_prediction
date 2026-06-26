# ⚽ World Cup 2026 — Match Predictor

Sistema end-to-end di Machine Learning che predice il risultato e il punteggio delle partite dei Mondiali FIFA 2026, a partire da dati storici reali e dalla forma aggiornata delle squadre.

🔗 **Live app:** _<inserisci URL Netlify>_
💻 **API:** _<inserisci URL Render>_

---

## 📌 Cosa fa

L'utente seleziona due nazionali e il modello restituisce:

- **Esito previsto** — vittoria squadra 1 / pareggio / vittoria squadra 2
- **Probabilità** per ciascun esito
- **Punteggio più probabile** (es. 2-0, 1-1, 3-1)
- **Gol attesi** per ogni squadra
- **Dettagli** — ranking FIFA, punti nel girone WC 2026, win rate storico, scontri diretti

L'interfaccia è disponibile in 6 lingue (IT, EN, FR, ES, DE, AR con supporto RTL).

---

## 🏗️ Architettura

```
Frontend (HTML/JS, Netlify)
        │  HTTP / JSON
        ▼
Backend (Flask API, Render)
        │
        ▼
Modello ML (Poisson + 2 regressori Random Forest)
```

---

## 🧠 Approccio del modello

Il punteggio di una partita è modellato con la **distribuzione di Poisson**, un approccio consolidato nella sport analytics:

1. Due regressori Random Forest stimano i **gol attesi** (lambda) di ciascuna squadra.
2. La distribuzione di Poisson costruisce la **matrice di probabilità** di ogni possibile risultato.
3. Da questa matrice si derivano in modo coerente sia l'esito (1/X/2) sia il punteggio più probabile.

Questo evita le contraddizioni tipiche di un classificatore e un regressore separati (es. "pareggio" come esito ma "2-1" come punteggio).

**Performance:** ROC-AUC ≈ 0.74 sulla classificazione dell'esito. Un valore onesto per il calcio, dove l'imprevedibilità è parte del gioco.

---

## 📊 I dati (data fusion)

Il dataset di training unisce tre fonti diverse:

| Fonte | Contenuto |
|-------|-----------|
| [martj42/international_results](https://github.com/martj42/international_results) | ~49.000 partite internazionali dal 1872 |
| Ranking FIFA | Classifica e punti aggiornati a giugno 2026 |
| Dataset WC 2026 | Forma delle squadre nel girone in corso |

### Feature engineering

Le feature più predittive sono state costruite a mano:

- **Forma recente** — win rate, gol segnati/subiti sugli ultimi 20 match
- **Differenza di ranking** e di punti FIFA tra le due squadre
- **Scontri diretti** (head-to-head) storici
- **Forma nel girone WC 2026** — punti e differenza reti

---

## 🗂️ Struttura del progetto

```
.
├── data/                   # dataset CSV (storico + WC 2026)
├── models/                 # modelli addestrati (.joblib)
├── clean.py                # pulizia + costruzione dataset
├── train.py                # training dei modelli
├── predict.py              # logica di previsione (Poisson)
├── app.py                  # API Flask
├── index.html              # frontend web app
├── translations.json       # traduzioni multilingua
├── requirements.txt
└── render.yaml             # config deploy Render
```

---

## 🚀 Uso in locale

```bash
# 1. ambiente virtuale
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. dipendenze
pip install -r requirements.txt

# 3. (opzionale) ricostruisci dataset e modelli
python clean.py
python train.py

# 4. avvia l'API
python app.py
```

L'API risponde su `http://127.0.0.1:5000`.

### Esempio di chiamata

```bash
POST /predict
{
    "team1": "Morocco",
    "team2": "Portugal"
}
```

Per il frontend, servi la cartella con un server statico (necessario per caricare `translations.json`):

```bash
python -m http.server 8000
```

---

## 🛠️ Stack tecnico

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Flask` · `Render` · `Netlify` · `HTML/CSS/JS`

---

## 📍 Endpoint API

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| `POST` | `/predict` | Previsione di una partita |
| `GET`  | `/teams`   | Lista delle squadre WC 2026 |
| `GET`  | `/health`  | Stato del servizio |

---

## ⚠️ Nota

Progetto a scopo didattico e dimostrativo. Le previsioni si basano su pattern statistici storici e non tengono conto di infortuni, motivazione, condizioni del giorno e di tutta l'imprevedibilità che rende il calcio quello che è.
