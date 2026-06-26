# ⚽ World Cup 2026 — Match Predictor

An end-to-end Machine Learning system that predicts the outcome and scoreline of FIFA World Cup 2026 matches, using real historical data and up-to-date team form.

🔗 **Live app:** _<your Netlify URL>_
💻 **API:** _<your Render URL>_

---

## 📌 What it does

The user selects two national teams and the model returns:

- **Predicted outcome** — team 1 win / draw / team 2 win
- **Probabilities** for each outcome
- **Most likely scoreline** (e.g. 2-0, 1-1, 3-1)
- **Expected goals** for each team
- **Details** — FIFA ranking, WC 2026 group points, historical win rate, head-to-head

The interface is available in 6 languages (IT, EN, FR, ES, DE, AR with RTL support).

---

## 🏗️ Architecture

```
Frontend (HTML/JS, Netlify)
        │  HTTP / JSON
        ▼
Backend (Flask API, Render)
        │
        ▼
ML model (Poisson + 2 Random Forest regressors)
```

---

## 🧠 Modeling approach

A match scoreline is modeled with the **Poisson distribution**, a well-established approach in sports analytics:

1. Two Random Forest regressors estimate each team's **expected goals** (lambda).
2. The Poisson distribution builds the **probability matrix** of every possible scoreline.
3. From this matrix, both the outcome (W/D/L) and the most likely scoreline are derived consistently.

This avoids the contradictions you get from a separate classifier and regressor (e.g. "draw" as the outcome but "2-1" as the scoreline).

**Performance:** ROC-AUC ≈ 0.74 on outcome classification. An honest figure for football, where unpredictability is part of the game.

---

## 📊 The data (data fusion)

The training dataset merges three different sources:

| Source | Content |
|--------|---------|
| [martj42/international_results](https://github.com/martj42/international_results) | ~49,000 international matches since 1872 |
| FIFA ranking | Rankings and points updated to June 2026 |
| WC 2026 dataset | Team form in the ongoing group stage |

### Feature engineering

The most predictive features were hand-built:

- **Recent form** — win rate, goals scored/conceded over the last 20 matches
- **Ranking and FIFA points difference** between the two teams
- **Head-to-head** historical record
- **WC 2026 group form** — points and goal difference

---

## 🗂️ Project structure

```
.
├── data/                   # CSV datasets (historical + WC 2026)
├── models/                 # trained models (.joblib)
├── clean.py                # cleaning + dataset construction
├── train.py                # model training
├── predict.py              # prediction logic (Poisson)
├── app.py                  # Flask API
├── index.html              # frontend web app
├── translations.json       # multilingual translations
├── requirements.txt
└── render.yaml             # Render deploy config
```

---

## 🚀 Running locally

```bash
# 1. virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. dependencies
pip install -r requirements.txt

# 3. (optional) rebuild dataset and models
python clean.py
python train.py

# 4. start the API
python app.py
```

The API runs on `http://127.0.0.1:5000`.

### Example request

```bash
POST /predict
{
    "team1": "Morocco",
    "team2": "Portugal"
}
```

For the frontend, serve the folder with a static server (needed to load `translations.json`):

```bash
python -m http.server 8000
```

---

## 🛠️ Tech stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Flask` · `Render` · `Netlify` · `HTML/CSS/JS`

---

## 📍 API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict a match |
| `GET`  | `/teams`   | List of WC 2026 teams |
| `GET`  | `/health`  | Service status |

---

## ⚠️ Note

Educational and demonstration project. Predictions are based on historical statistical patterns and do not account for injuries, motivation, day-to-day conditions, or all the unpredictability that makes football what it is.

# IT

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
