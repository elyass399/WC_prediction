"""
Previsione di una singola partita del Mondiale 2026.

Pipeline di previsione:
    1. Calcola le feature delle due squadre (forma, head-to-head, ranking,
       valore di mercato, esperienza WC, rendimento qualificazioni 2026).
    2. I due regressori RandomForest stimano i gol attesi (lambda1, lambda2).
    3. Un modello di Poisson trasforma i lambda in probabilità di esito e in
       un risultato (score) coerente con l'esito più probabile.

I dati statici e i modelli vengono caricati una sola volta e tenuti in cache,
così le richieste successive (es. dall'API Flask) sono veloci.
"""
from __future__ import annotations

from math import exp, factorial
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
MAX_GOL = 8   # numero massimo di gol per squadra considerato nella matrice Poisson

# Ordine delle feature: deve coincidere esattamente con quello usato in train.py
FEATURES = [
    "team1_win_rate",        "team2_win_rate",
    "team1_draw_rate",       "team2_draw_rate",
    "team1_loss_rate",       "team2_loss_rate",
    "team1_gol_segnati_avg", "team2_gol_segnati_avg",
    "team1_gol_subiti_avg",  "team2_gol_subiti_avg",
    "team1_win_rate_uff",    "team2_win_rate_uff",
    "team1_n_partite",       "team2_n_partite",
    "h2h_team1_wins", "h2h_draws", "h2h_n",
    "team1_fifa_rank",    "team2_fifa_rank",
    "team1_fifa_points",  "team2_fifa_points",
    "rank_diff",          "points_diff",
    "team1_wc_titles", "team2_wc_titles",
    "team1_wc_exp",    "team2_wc_exp",
    "team1_market_value", "team2_market_value",
    "team1_wc2026_punti",    "team2_wc2026_punti",
    "team1_wc2026_diff_gol", "team2_wc2026_diff_gol",
]

# Normalizzazione dei nomi nazionali (deve restare allineata a clean.py)
NOME_FIX = {
    "Czechia":       "Czech Republic",
    "Congo DR":      "DR Congo",
    "USA":           "United States",
    "IR Iran":       "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye":       "Turkey",
    "Cabo Verde":    "Cape Verde",
}

NOME_FIX_WC = {
    "IR Iran":       "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye":       "Turkey",
    "Cabo Verde":    "Cape Verde",
}

TORNEI_UFFICIALI = {
    "FIFA World Cup", "FIFA World Cup qualification",
    "UEFA Euro", "UEFA Euro qualification", "UEFA Nations League",
    "Copa América", "African Cup of Nations",
    "African Cup of Nations qualification",
    "AFC Asian Cup", "AFC Asian Cup qualification",
    "Gold Cup", "CONCACAF Nations League", "CONCACAF Championship",
}


# ─────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────
def _primo_valore(df_filtrato: pd.DataFrame, colonna: str, default):
    """Primo valore (float) della colonna, o `default` se il filtro è vuoto."""
    if len(df_filtrato) > 0:
        return float(df_filtrato[colonna].values[0])
    return default


# ─────────────────────────────────────────
# CACHE MODELLI
# ─────────────────────────────────────────
_CACHE: Dict[str, Any] = {}


def carica_modello(nome: str) -> Dict[str, Any]:
    """Carica un modello .joblib dalla cartella models/, con cache in memoria."""
    if nome not in _CACHE:
        path = MODELS_DIR / nome
        if not path.exists():
            raise FileNotFoundError(f"Modello non trovato: {path}")
        _CACHE[nome] = joblib.load(path)
    return _CACHE[nome]


# ─────────────────────────────────────────
# DATI STATICI (caricati una sola volta)
# ─────────────────────────────────────────
_RESULTS = None   # storico partite (results.csv)
_FIFA = None      # ranking FIFA
_TRAIN = None     # valore mercato + esperienza WC
_WC2026 = None    # rendimento qualificazioni WC 2026


def carica_dati_statici():
    """Carica e prepara i dati condivisi, popolando le cache globali se vuote."""
    global _RESULTS, _FIFA, _TRAIN, _WC2026

    # Storico partite con esito calcolato
    if _RESULTS is None:
        df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
        df["home_team"] = df["home_team"].replace(NOME_FIX)
        df["away_team"] = df["away_team"].replace(NOME_FIX)
        df = df.dropna(subset=["home_score", "away_score"])
        df["result"] = np.where(
            df["home_score"] > df["away_score"], "H",
            np.where(df["home_score"] < df["away_score"], "A", "D")
        )
        _RESULTS = df

    # Ranking FIFA
    if _FIFA is None:
        _FIFA = pd.read_csv(DATA_DIR / "fifa_rankings_2026.csv")

    # Dati di squadra (solo la versione più recente per ciascun team)
    if _TRAIN is None:
        train = pd.read_csv(DATA_DIR / "train.csv")
        train_latest = train.sort_values("version").groupby("team").last().reset_index()
        train_latest["team_fix"] = train_latest["team"].replace(NOME_FIX)
        _TRAIN = train_latest

    # Rendimento nelle qualificazioni WC 2026 (punti e differenza reti)
    if _WC2026 is None:
        df_wc = pd.read_csv(DATA_DIR / "matches_detailed.csv")
        df_wc["home_team_name"] = df_wc["home_team_name"].replace(NOME_FIX_WC)
        df_wc["away_team_name"] = df_wc["away_team_name"].replace(NOME_FIX_WC)
        df_wc = df_wc[df_wc["status"] == "Completed"].dropna(subset=["home_score", "away_score"])

        squadre = set(df_wc["home_team_name"].tolist() + df_wc["away_team_name"].tolist())
        righe = []
        for team in squadre:
            partite_casa = df_wc[df_wc["home_team_name"] == team]
            partite_trasferta = df_wc[df_wc["away_team_name"] == team]
            punti = gol_fatti = gol_subiti = 0

            for _, r in partite_casa.iterrows():
                gf, gs = r["home_score"], r["away_score"]
                gol_fatti += gf
                gol_subiti += gs
                if gf > gs:
                    punti += 3
                elif gf == gs:
                    punti += 1

            for _, r in partite_trasferta.iterrows():
                gf, gs = r["away_score"], r["home_score"]
                gol_fatti += gf
                gol_subiti += gs
                if gf > gs:
                    punti += 3
                elif gf == gs:
                    punti += 1

            righe.append({
                "team":            team,
                "wc2026_punti":    punti,
                "wc2026_diff_gol": gol_fatti - gol_subiti,
            })
        _WC2026 = pd.DataFrame(righe)


# ─────────────────────────────────────────
# FEATURES SQUADRA
# ─────────────────────────────────────────
def calcola_features_squadra(team: str, n: int = 20) -> dict:
    """Statistiche di forma di `team` sulle sue ultime `n` partite disponibili."""
    df = _RESULTS
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    partite = df[mask].tail(n)

    # Nessuna partita nota: ritorna feature azzerate
    if len(partite) == 0:
        return {
            "win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0,
            "gol_segnati_avg": 0.0, "gol_subiti_avg": 0.0,
            "win_rate_uff": 0.0, "n_partite": 0,
        }

    vittorie = pareggi = sconfitte = 0
    gol_segnati = gol_subiti = 0
    vittorie_uff = partite_uff = 0

    for _, row in partite.iterrows():
        gioca_in_casa = row["home_team"] == team
        gs = row["home_score"] if gioca_in_casa else row["away_score"]
        gc = row["away_score"] if gioca_in_casa else row["home_score"]

        esito = row["result"]
        ha_vinto = (esito == "H") if gioca_in_casa else (esito == "A")
        ha_pareggiato = esito == "D"

        gol_segnati += gs
        gol_subiti += gc
        if ha_vinto:
            vittorie += 1
        elif ha_pareggiato:
            pareggi += 1
        else:
            sconfitte += 1

        # Conteggio separato per le partite ufficiali
        if row["tournament"] in TORNEI_UFFICIALI:
            partite_uff += 1
            if ha_vinto:
                vittorie_uff += 1

    n_tot = len(partite)
    return {
        "win_rate":        round(vittorie / n_tot, 4),
        "draw_rate":       round(pareggi / n_tot, 4),
        "loss_rate":       round(sconfitte / n_tot, 4),
        "gol_segnati_avg": round(gol_segnati / n_tot, 4),
        "gol_subiti_avg":  round(gol_subiti / n_tot, 4),
        "win_rate_uff":    round(vittorie_uff / partite_uff, 4) if partite_uff > 0 else 0.0,
        "n_partite":       n_tot,
    }


def calcola_h2h(team1: str, team2: str) -> dict:
    """Statistiche degli scontri diretti dal punto di vista di team1."""
    df = _RESULTS
    mask = (
        ((df["home_team"] == team1) & (df["away_team"] == team2)) |
        ((df["home_team"] == team2) & (df["away_team"] == team1))
    )
    h2h = df[mask]

    if len(h2h) == 0:
        return {"h2h_team1_wins": 0.0, "h2h_draws": 0.0, "h2h_n": 0}

    team1_wins = pareggi = 0
    for _, row in h2h.iterrows():
        if row["home_team"] == team1:
            if row["result"] == "H":
                team1_wins += 1
            elif row["result"] == "D":
                pareggi += 1
        else:
            if row["result"] == "A":
                team1_wins += 1
            elif row["result"] == "D":
                pareggi += 1

    n = len(h2h)
    return {
        "h2h_team1_wins": round(team1_wins / n, 4),
        "h2h_draws":      round(pareggi / n, 4),
        "h2h_n":          n,
    }


# ─────────────────────────────────────────
# MODELLO DI POISSON
# ─────────────────────────────────────────
def poisson_pmf(k: int, lam: float) -> float:
    """Probabilità di osservare esattamente `k` gol con media attesa `lam`."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * exp(-lam) / factorial(k)


def matrice_poisson(lambda1: float, lambda2: float) -> Dict[str, Any]:
    """Dai gol attesi (lambda) calcola probabilità di esito e risultato previsto.

    Si costruisce la matrice di tutti i punteggi possibili (0..MAX_GOL per
    squadra) e si sommano le probabilità per esito (vittoria team1 / pareggio /
    vittoria team2). Il verdetto è l'esito con probabilità totale più alta; lo
    score mostrato è la cella più probabile DENTRO quell'esito, così risultato
    e punteggio sono sempre coerenti tra loro.
    """
    # Limita i lambda a un intervallo plausibile per stabilità numerica
    lambda1 = max(0.1, min(lambda1, 6.0))
    lambda2 = max(0.1, min(lambda2, 6.0))

    p_team1 = p_draw = p_team2 = 0.0
    # Per ogni esito teniamo la cella più probabile: (probabilità, (gol1, gol2))
    best = {"team1": (-1.0, (1, 0)), "draw": (-1.0, (1, 1)), "team2": (-1.0, (0, 1))}

    for g1 in range(MAX_GOL + 1):
        for g2 in range(MAX_GOL + 1):
            p = poisson_pmf(g1, lambda1) * poisson_pmf(g2, lambda2)
            if g1 > g2:
                p_team1 += p
                if p > best["team1"][0]:
                    best["team1"] = (p, (g1, g2))
            elif g1 < g2:
                p_team2 += p
                if p > best["team2"][0]:
                    best["team2"] = (p, (g1, g2))
            else:
                p_draw += p
                if p > best["draw"][0]:
                    best["draw"] = (p, (g1, g2))

    # Normalizza le probabilità (la matrice è troncata a MAX_GOL, somma < 1)
    tot = p_team1 + p_draw + p_team2
    if tot > 0:
        p_team1 /= tot
        p_draw /= tot
        p_team2 /= tot

    # Verdetto = esito più probabile, score = cella migliore di quell'esito
    probs = {"team1": p_team1, "draw": p_draw, "team2": p_team2}
    esito = max(probs, key=probs.get)
    score = best[esito][1]

    return {
        "p_team1_win": round(p_team1, 4),
        "p_draw":      round(p_draw, 4),
        "p_team2_win": round(p_team2, 4),
        "score":       score,
        "esito":       esito,
        "lambda1":     round(lambda1, 2),
        "lambda2":     round(lambda2, 2),
    }


# ─────────────────────────────────────────
# FUNZIONE PRINCIPALE
# ─────────────────────────────────────────
def previsione_partita(team1: str, team2: str) -> Dict[str, Any]:
    """Previsione completa della partita team1 vs team2."""
    carica_dati_statici()

    # Normalizza i nomi in ingresso al formato canonico dei dataset
    team1_fix = NOME_FIX.get(team1, team1)
    team2_fix = NOME_FIX.get(team2, team2)

    # Feature di forma e scontri diretti
    f1 = calcola_features_squadra(team1_fix)
    f2 = calcola_features_squadra(team2_fix)
    h2h = calcola_h2h(team1_fix, team2_fix)

    # Ranking FIFA e relative differenze
    r1 = _FIFA[_FIFA["team"] == team1_fix]
    r2 = _FIFA[_FIFA["team"] == team2_fix]
    t1_rank = _primo_valore(r1, "fifa_rank", np.nan)
    t2_rank = _primo_valore(r2, "fifa_rank", np.nan)
    t1_points = _primo_valore(r1, "fifa_points", np.nan)
    t2_points = _primo_valore(r2, "fifa_points", np.nan)
    rank_diff = (t1_rank - t2_rank) if not (np.isnan(t1_rank) or np.isnan(t2_rank)) else np.nan
    points_diff = (t1_points - t2_points) if not (np.isnan(t1_points) or np.isnan(t2_points)) else np.nan

    # Valore di mercato ed esperienza ai Mondiali
    tr1 = _TRAIN[_TRAIN["team_fix"] == team1_fix]
    tr2 = _TRAIN[_TRAIN["team_fix"] == team2_fix]
    t1_mv = _primo_valore(tr1, "squad_total_market_value_eur", np.nan)
    t2_mv = _primo_valore(tr2, "squad_total_market_value_eur", np.nan)
    t1_titles = _primo_valore(tr1, "world_cup_titles_before", 0)
    t2_titles = _primo_valore(tr2, "world_cup_titles_before", 0)
    t1_wcp = _primo_valore(tr1, "world_cup_participations_before", 0)
    t2_wcp = _primo_valore(tr2, "world_cup_participations_before", 0)

    # Rendimento nelle qualificazioni WC 2026
    wc1 = _WC2026[_WC2026["team"] == team1_fix]
    wc2 = _WC2026[_WC2026["team"] == team2_fix]
    t1_wc_punti = _primo_valore(wc1, "wc2026_punti", np.nan)
    t2_wc_punti = _primo_valore(wc2, "wc2026_punti", np.nan)
    t1_wc_diff = _primo_valore(wc1, "wc2026_diff_gol", np.nan)
    t2_wc_diff = _primo_valore(wc2, "wc2026_diff_gol", np.nan)

    # Riga di input con tutte le feature attese dai modelli
    input_data = {
        "team1_win_rate": f1["win_rate"], "team2_win_rate": f2["win_rate"],
        "team1_draw_rate": f1["draw_rate"], "team2_draw_rate": f2["draw_rate"],
        "team1_loss_rate": f1["loss_rate"], "team2_loss_rate": f2["loss_rate"],
        "team1_gol_segnati_avg": f1["gol_segnati_avg"], "team2_gol_segnati_avg": f2["gol_segnati_avg"],
        "team1_gol_subiti_avg": f1["gol_subiti_avg"], "team2_gol_subiti_avg": f2["gol_subiti_avg"],
        "team1_win_rate_uff": f1["win_rate_uff"], "team2_win_rate_uff": f2["win_rate_uff"],
        "team1_n_partite": f1["n_partite"], "team2_n_partite": f2["n_partite"],
        "h2h_team1_wins": h2h["h2h_team1_wins"], "h2h_draws": h2h["h2h_draws"], "h2h_n": h2h["h2h_n"],
        "team1_fifa_rank": t1_rank, "team2_fifa_rank": t2_rank,
        "team1_fifa_points": t1_points, "team2_fifa_points": t2_points,
        "rank_diff": rank_diff, "points_diff": points_diff,
        "team1_wc_titles": t1_titles, "team2_wc_titles": t2_titles,
        "team1_wc_exp": t1_wcp, "team2_wc_exp": t2_wcp,
        "team1_market_value": t1_mv, "team2_market_value": t2_mv,
        "team1_wc2026_punti": t1_wc_punti, "team2_wc2026_punti": t2_wc_punti,
        "team1_wc2026_diff_gol": t1_wc_diff, "team2_wc2026_diff_gol": t2_wc_diff,
    }
    input_df = pd.DataFrame([input_data], columns=FEATURES)

    # Regressori → gol attesi (lambda) per ciascuna squadra
    lambda1 = float(carica_modello("modello_gol_team1.joblib")["pipeline"].predict(input_df)[0])
    lambda2 = float(carica_modello("modello_gol_team2.joblib")["pipeline"].predict(input_df)[0])

    # Poisson → probabilità di esito e score coerente
    poi = matrice_poisson(lambda1, lambda2)
    gol1, gol2 = poi["score"]

    # Traduce l'esito interno nelle etichette esposte all'esterno
    if poi["esito"] == "team1":
        risultato = "team1_win"
    elif poi["esito"] == "team2":
        risultato = "team2_win"
    else:
        risultato = "draw"

    return {
        "team1": team1,
        "team2": team2,
        "risultato": risultato,
        "probabilita": {
            "team1_win": poi["p_team1_win"],
            "draw":      poi["p_draw"],
            "team2_win": poi["p_team2_win"],
        },
        "score_previsto": f"{gol1}-{gol2}",
        "gol_team1": gol1,
        "gol_team2": gol2,
        "gol_attesi": {"team1": poi["lambda1"], "team2": poi["lambda2"]},
        "dettagli": {
            "team1_fifa_rank":    t1_rank,
            "team2_fifa_rank":    t2_rank,
            "team1_wc2026_punti": t1_wc_punti,
            "team2_wc2026_punti": t2_wc_punti,
            "team1_win_rate":     f1["win_rate"],
            "team2_win_rate":     f2["win_rate"],
            "h2h_partite_totali": h2h["h2h_n"],
        },
    }


# ─────────────────────────────────────────
# TEST MANUALE
# ─────────────────────────────────────────
if __name__ == "__main__":
    partite_di_prova = [
        ("Iran", "England"),
        ("Morocco", "Netherlands"),
        ("Bosnia and Herzegovina", "Qatar"),
        ("Brazil", "Argentina"),
        ("Germany", "Curaçao"),
        ("Spain", "Haiti"),
    ]
    for t1, t2 in partite_di_prova:
        r = previsione_partita(t1, t2)
        print(f"\n=== {t1} vs {t2} ===")
        print(f"  Score: {r['score_previsto']}  ({r['risultato']})")
        print(f"  Gol attesi: {r['gol_attesi']['team1']} - {r['gol_attesi']['team2']}")
        print(f"  Prob: T1 {r['probabilita']['team1_win']} | "
              f"X {r['probabilita']['draw']} | T2 {r['probabilita']['team2_win']}")
