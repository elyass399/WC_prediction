"""
Script di ispezione rapida del file delle partite del Mondiale 2026.

Stampa le dimensioni del dataset e le colonne principali (squadre,
punteggi e stato della partita) per controllare al volo i dati grezzi.
"""
import pandas as pd

# Carica le partite dettagliate
df = pd.read_csv("data/matches_detailed.csv")

# Mostra dimensioni (righe, colonne) e un estratto delle colonne chiave
print(df.shape)
print(df[["home_team_name", "away_team_name", "home_score", "away_score", "status"]].to_string())
