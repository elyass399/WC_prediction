
import math

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from predict import previsione_partita


# CONFIGURAZIONE APP

app = Flask(__name__)
CORS(app)  # abilita le richieste cross-origin (es. dal frontend)

TEAMS_CSV = "data/teams.csv"





def sanitizza(obj):
    """Sostituisce NaN/Infinity con None così da produrre JSON valido.

    Il JSON standard non ammette NaN/Infinity: senza questa pulizia il
    client riceverebbe un payload non parsabile. La funzione è ricorsiva
    per gestire dizionari e liste annidate.
    """
    if isinstance(obj, dict):
        return {chiave: sanitizza(valore) for chiave, valore in obj.items()}
    if isinstance(obj, list):
        return [sanitizza(valore) for valore in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ENDPOINTS

@app.route("/predict", methods=["POST"])
def predict():
    """Riceve {team1, team2} in JSON e restituisce la previsione della partita."""
    # 1. Il body deve essere JSON valido
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"errore": "Body JSON mancante"}), 400

    # 2. Entrambe le squadre devono essere presenti
    if "team1" not in data or "team2" not in data:
        return jsonify({
            "errore": "Campi mancanti",
            "richiesti": ["team1", "team2"],
        }), 400

    team1 = str(data["team1"]).strip()
    team2 = str(data["team2"]).strip()

    # 3. Le due squadre devono essere diverse
    if team1.lower() == team2.lower():
        return jsonify({"errore": "team1 e team2 devono essere squadre diverse"}), 400

    # 4. Esegue la previsione tramite il modello
    try:
        risultato = previsione_partita(team1, team2)
    except Exception as errore:
        return jsonify({"errore": str(errore)}), 500

    return jsonify(sanitizza(risultato)), 200


@app.route("/teams", methods=["GET"])
def teams():
    """Restituisce la lista ordinata delle squadre del Mondiale 2026."""
    teams_df = pd.read_csv(TEAMS_CSV)
    lista = sorted(teams_df["team_name"].tolist())
    return jsonify({"squadre": lista, "totale": len(lista)}), 200


@app.route("/health", methods=["GET"])
def health():
    """Verifica veloce dello stato del servizio."""
    return jsonify({
        "status": "ok",
        "modello": "Poisson (2 regressori RandomForest)",
        "files": [
            "modello_gol_team1.joblib",
            "modello_gol_team2.joblib",
        ],
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
