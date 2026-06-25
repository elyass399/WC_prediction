
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT = "cleaned_data.csv"
DATE_START = "2010-01-01"   # consideriamo solo le partite dal 2010 in poi
N_MATCH = 20                # numero di partite recenti usate per la "forma"

# Normalizzazione dei nomi nazionali in results.csv (varianti → nome canonico)
NOME_FIX = {
    "Czechia":       "Czech Republic",
    "Congo DR":      "DR Congo",
    "USA":           "United States",
    "IR Iran":       "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye":       "Turkey",
    "Cabo Verde":    "Cape Verde",
}

# Normalizzazione dei nomi usati nel file delle partite WC 2026
NOME_FIX_WC = {
    "IR Iran":       "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye":       "Turkey",
    "Cabo Verde":    "Cape Verde",
}

# Tornei considerati "ufficiali" (rilevanti per la qualità della forma)
TORNEI_UFFICIALI = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "UEFA Nations League",
    "Copa América",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "Gold Cup",
    "CONCACAF Nations League",
    "CONCACAF Championship",
}


# ─────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────
def _primo_valore(df_filtrato: pd.DataFrame, colonna: str, default):
    """Restituisce il primo valore della colonna, o `default` se vuoto."""
    if len(df_filtrato) > 0:
        return df_filtrato[colonna].values[0]
    return default


# ─────────────────────────────────────────
# STEP 1 — Carica e prepara results.csv
# ─────────────────────────────────────────
def carica_results() -> pd.DataFrame:
    """Carica lo storico partite, normalizza i nomi e calcola colonne derivate."""
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])

    # Uniforma i nomi delle squadre
    df["home_team"] = df["home_team"].replace(NOME_FIX)
    df["away_team"] = df["away_team"].replace(NOME_FIX)

    # Scarta le partite senza punteggio (non utilizzabili)
    df = df.dropna(subset=["home_score", "away_score"])

    # Esito dal punto di vista della squadra di casa: H = casa, A = trasferta, D = pari
    df["result"] = np.where(
        df["home_score"] > df["away_score"], "H",
        np.where(df["home_score"] < df["away_score"], "A", "D")
    )

    # Flag torneo ufficiale
    df["is_official"] = df["tournament"].isin(TORNEI_UFFICIALI)

    print(f"✅ results.csv: {len(df)} partite dopo drop NaN")
    return df.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────
# STEP 2 — Features per squadra (forma recente)
# ─────────────────────────────────────────
def features_squadra(team: str, before_date: pd.Timestamp,
                     df: pd.DataFrame, n: int = N_MATCH) -> dict:
    """Calcola le statistiche di forma di `team` sulle ultime `n` partite
    giocate PRIMA di `before_date` (per evitare data leakage)."""
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    )
    partite = df[mask].tail(n)

    if len(partite) == 0:
        return _features_vuote(team)

    vittorie = pareggi = sconfitte = 0
    gol_segnati = gol_subiti = 0
    vittorie_uff = partite_uff = 0

    for _, row in partite.iterrows():
        gioca_in_casa = row["home_team"] == team
        # Gol segnati/subiti dal punto di vista di `team`
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

        # Conteggio separato per le sole partite ufficiali
        if row["is_official"]:
            partite_uff += 1
            if ha_vinto:
                vittorie_uff += 1

    n_tot = len(partite)
    return {
        f"{team}_win_rate":        round(vittorie / n_tot, 4),
        f"{team}_draw_rate":       round(pareggi / n_tot, 4),
        f"{team}_loss_rate":       round(sconfitte / n_tot, 4),
        f"{team}_gol_segnati_avg": round(gol_segnati / n_tot, 4),
        f"{team}_gol_subiti_avg":  round(gol_subiti / n_tot, 4),
        f"{team}_win_rate_uff":    round(vittorie_uff / partite_uff, 4) if partite_uff > 0 else 0.0,
        f"{team}_n_partite":       n_tot,
    }


def _features_vuote(team: str) -> dict:
    """Valori di default quando non ci sono partite precedenti per `team`."""
    return {
        f"{team}_win_rate":        0.0,
        f"{team}_draw_rate":       0.0,
        f"{team}_loss_rate":       0.0,
        f"{team}_gol_segnati_avg": 0.0,
        f"{team}_gol_subiti_avg":  0.0,
        f"{team}_win_rate_uff":    0.0,
        f"{team}_n_partite":       0,
    }


# ─────────────────────────────────────────
# STEP 3 — Head to head (scontri diretti)
# ─────────────────────────────────────────
def head_to_head(team1: str, team2: str,
                 before_date: pd.Timestamp,
                 df: pd.DataFrame) -> dict:
    """Statistiche degli scontri diretti tra `team1` e `team2` prima di `before_date`."""
    mask = (
        (
            ((df["home_team"] == team1) & (df["away_team"] == team2)) |
            ((df["home_team"] == team2) & (df["away_team"] == team1))
        ) &
        (df["date"] < before_date)
    )
    h2h = df[mask]

    if len(h2h) == 0:
        return {"h2h_team1_wins": 0.0, "h2h_draws": 0.0, "h2h_n": 0}

    team1_wins = pareggi = 0
    for _, row in h2h.iterrows():
        # Riconduce sempre l'esito al punto di vista di team1
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
# STEP 4 — Carica info squadre (FIFA + train.csv)
# ─────────────────────────────────────────
def carica_team_info():
    """Carica il ranking FIFA e i dati di squadra (versione più recente per team)."""
    fifa = pd.read_csv(DATA_DIR / "fifa_rankings_2026.csv")

    train = pd.read_csv(DATA_DIR / "train.csv")
    # Per ogni squadra teniamo solo la riga con la versione più alta (la più recente)
    train_latest = (
        train.sort_values("version")
        .groupby("team")
        .last()
        .reset_index()
    )
    train_latest["team_fix"] = train_latest["team"].replace(NOME_FIX)
    return fifa, train_latest


# ─────────────────────────────────────────
# STEP 5 — Features dalle qualificazioni WC 2026
# ─────────────────────────────────────────
def calcola_features_wc2026() -> pd.DataFrame:
    """Calcola punti e differenza reti di ogni squadra nelle partite WC 2026 completate."""
    df = pd.read_csv(DATA_DIR / "matches_detailed.csv")
    df["home_team_name"] = df["home_team_name"].replace(NOME_FIX_WC)
    df["away_team_name"] = df["away_team_name"].replace(NOME_FIX_WC)

    # Solo partite concluse con punteggio valido
    df = df[df["status"] == "Completed"].copy()
    df = df.dropna(subset=["home_score", "away_score"])

    squadre = set(df["home_team_name"].tolist() + df["away_team_name"].tolist())
    righe = []

    for team in squadre:
        partite_casa = df[df["home_team_name"] == team]
        partite_trasferta = df[df["away_team_name"] == team]

        punti = gol_fatti = gol_subiti = 0

        # Partite in casa
        for _, row in partite_casa.iterrows():
            gf, gs = row["home_score"], row["away_score"]
            gol_fatti += gf
            gol_subiti += gs
            if gf > gs:
                punti += 3
            elif gf == gs:
                punti += 1

        # Partite in trasferta
        for _, row in partite_trasferta.iterrows():
            gf, gs = row["away_score"], row["home_score"]
            gol_fatti += gf
            gol_subiti += gs
            if gf > gs:
                punti += 3
            elif gf == gs:
                punti += 1

        n = len(partite_casa) + len(partite_trasferta)
        righe.append({
            "team":              team,
            "wc2026_punti":      punti,
            "wc2026_gol_fatti":  gol_fatti,
            "wc2026_gol_subiti": gol_subiti,
            "wc2026_diff_gol":   gol_fatti - gol_subiti,
            "wc2026_partite":    n,
        })

    df_wc = pd.DataFrame(righe)
    print(f"\n✅ WC 2026 features: {len(df_wc)} squadre")
    print(df_wc.sort_values("wc2026_punti", ascending=False).to_string(index=False))
    return df_wc


# ─────────────────────────────────────────
# STEP 6 — Costruisci il dataset finale
# ─────────────────────────────────────────
def costruisci_dataset(df: pd.DataFrame,
                       fifa: pd.DataFrame,
                       train: pd.DataFrame,
                       wc2026: pd.DataFrame) -> pd.DataFrame:
    """Combina tutte le fonti in una riga per partita ufficiale dal 2010."""
    df_match = df[
        (df["date"] >= DATE_START) &
        (df["is_official"])
    ].copy()

    print(f"\n✅ Partite ufficiali dal 2010: {len(df_match)}")

    righe = []

    for i, row in df_match.iterrows():
        team1 = row["home_team"]
        team2 = row["away_team"]
        date = row["date"]

        # Forma recente: le chiavi contengono il nome squadra, le rinominiamo
        # in "team1"/"team2" per avere colonne uniformi nel dataset.
        f1 = features_squadra(team1, date, df)
        f2 = features_squadra(team2, date, df)
        f1 = {k.replace(team1, "team1"): v for k, v in f1.items()}
        f2 = {k.replace(team2, "team2"): v for k, v in f2.items()}
        h2h = head_to_head(team1, team2, date, df)

        # Ranking FIFA e relative differenze
        r1 = fifa[fifa["team"] == team1]
        r2 = fifa[fifa["team"] == team2]
        t1_fifa = _primo_valore(r1, "fifa_rank", np.nan)
        t2_fifa = _primo_valore(r2, "fifa_rank", np.nan)
        t1_points = _primo_valore(r1, "fifa_points", np.nan)
        t2_points = _primo_valore(r2, "fifa_points", np.nan)
        rank_diff = (t1_fifa - t2_fifa) if not (np.isnan(t1_fifa) or np.isnan(t2_fifa)) else np.nan
        points_diff = (t1_points - t2_points) if not (np.isnan(t1_points) or np.isnan(t2_points)) else np.nan

        # Valore di mercato ed esperienza ai Mondiali
        tr1 = train[train["team_fix"] == team1]
        tr2 = train[train["team_fix"] == team2]
        t1_mv = _primo_valore(tr1, "squad_total_market_value_eur", np.nan)
        t2_mv = _primo_valore(tr2, "squad_total_market_value_eur", np.nan)
        t1_titles = _primo_valore(tr1, "world_cup_titles_before", 0)
        t2_titles = _primo_valore(tr2, "world_cup_titles_before", 0)
        t1_wcp = _primo_valore(tr1, "world_cup_participations_before", 0)
        t2_wcp = _primo_valore(tr2, "world_cup_participations_before", 0)

        # Rendimento nelle qualificazioni WC 2026
        wc1 = wc2026[wc2026["team"] == team1]
        wc2 = wc2026[wc2026["team"] == team2]
        t1_wc_punti = _primo_valore(wc1, "wc2026_punti", np.nan)
        t2_wc_punti = _primo_valore(wc2, "wc2026_punti", np.nan)
        t1_wc_diff = _primo_valore(wc1, "wc2026_diff_gol", np.nan)
        t2_wc_diff = _primo_valore(wc2, "wc2026_diff_gol", np.nan)

        riga = {
            "date":  date,
            "team1": team1,
            "team2": team2,
            **f1, **f2, **h2h,
            "team1_fifa_rank":       t1_fifa,
            "team2_fifa_rank":       t2_fifa,
            "team1_fifa_points":     t1_points,
            "team2_fifa_points":     t2_points,
            "rank_diff":             rank_diff,
            "points_diff":           points_diff,
            "team1_market_value":    t1_mv,
            "team2_market_value":    t2_mv,
            "team1_wc_titles":       t1_titles,
            "team2_wc_titles":       t2_titles,
            "team1_wc_exp":          t1_wcp,
            "team2_wc_exp":          t2_wcp,
            "team1_wc2026_punti":    t1_wc_punti,
            "team2_wc2026_punti":    t2_wc_punti,
            "team1_wc2026_diff_gol": t1_wc_diff,
            "team2_wc2026_diff_gol": t2_wc_diff,
            # Target del training
            "result":    row["result"],
            "gol_team1": row["home_score"],
            "gol_team2": row["away_score"],
        }
        righe.append(riga)

        # Log di avanzamento ogni 500 partite
        if (i + 1) % 500 == 0:
            print(f"  → {i + 1} partite processate...")

    return pd.DataFrame(righe)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main() -> None:
    print("=== Caricamento dati ===")
    df = carica_results()
    fifa, train = carica_team_info()
    wc2026 = calcola_features_wc2026()

    print("\n=== Costruzione dataset ===")
    dataset = costruisci_dataset(df, fifa, train, wc2026)

    print("\n=== Dataset finale ===")
    print(f"Shape: {dataset.shape}")
    print(f"\nDistribuzione risultati:\n{dataset['result'].value_counts()}")
    print(f"\nGol medi team1: {dataset['gol_team1'].mean():.2f}")
    print(f"Gol medi team2: {dataset['gol_team2'].mean():.2f}")
    nan_per_colonna = dataset.isnull().sum()
    print(f"\nNaN per colonna:\n{nan_per_colonna[nan_per_colonna > 0]}")

    dataset.to_csv(OUTPUT, index=False)
    print(f"\n✅ Salvato: {OUTPUT}")


if __name__ == "__main__":
    main()
