
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold, RandomizedSearchCV,
    StratifiedKFold, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# COSTANTI
# ─────────────────────────────────────────
CSV_INPUT = "cleaned_data.csv"
MODELS_DIR = Path("models")

# Colonne target
TARGET_CLASS = "result"
TARGET_GOL1 = "gol_team1"
TARGET_GOL2 = "gol_team2"

# Feature usate da tutti i modelli (devono coincidere con quelle di predict.py)
FEATURES = [
    # Forma storica recente
    "team1_win_rate",        "team2_win_rate",
    "team1_draw_rate",       "team2_draw_rate",
    "team1_loss_rate",       "team2_loss_rate",
    "team1_gol_segnati_avg", "team2_gol_segnati_avg",
    "team1_gol_subiti_avg",  "team2_gol_subiti_avg",
    "team1_win_rate_uff",    "team2_win_rate_uff",
    "team1_n_partite",       "team2_n_partite",
    # Scontri diretti
    "h2h_team1_wins", "h2h_draws", "h2h_n",
    # Ranking FIFA
    "team1_fifa_rank",    "team2_fifa_rank",
    "team1_fifa_points",  "team2_fifa_points",
    "rank_diff",          "points_diff",
    # Esperienza ai Mondiali
    "team1_wc_titles", "team2_wc_titles",
    "team1_wc_exp",    "team2_wc_exp",
    # Valore di mercato
    "team1_market_value", "team2_market_value",
    # Rendimento qualificazioni WC 2026
    "team1_wc2026_punti",    "team2_wc2026_punti",
    "team1_wc2026_diff_gol", "team2_wc2026_diff_gol",
]


# ─────────────────────────────────────────
# STEP 1 — Carica i dati
# ─────────────────────────────────────────
def carica_dati(path: str = CSV_INPUT) -> pd.DataFrame:
    """Carica il dataset prodotto da clean.py."""
    df = pd.read_csv(path)
    print(f"✅ Caricato: {df.shape[0]} righe, {df.shape[1]} colonne")
    return df


# ─────────────────────────────────────────
# STEP 2 — Analisi esplorativa
# ─────────────────────────────────────────
def analisi_esplorativa(df: pd.DataFrame) -> None:
    """Stampa distribuzione esiti, gol medi e NaN: utile per un sanity check iniziale."""
    print("\n=== Distribuzione risultati ===")
    vc = df[TARGET_CLASS].value_counts()
    print(vc)
    print(f"  H: {vc['H']/len(df)*100:.1f}% | D: {vc['D']/len(df)*100:.1f}% | A: {vc['A']/len(df)*100:.1f}%")

    print("\n=== Gol medi ===")
    print(f"  team1: {df[TARGET_GOL1].mean():.2f} ± {df[TARGET_GOL1].std():.2f}")
    print(f"  team2: {df[TARGET_GOL2].mean():.2f} ± {df[TARGET_GOL2].std():.2f}")

    print("\n=== NaN per feature ===")
    nan = df[FEATURES].isnull().sum()
    print(nan[nan > 0])


# ─────────────────────────────────────────
# STEP 3 — Preprocessore
# ─────────────────────────────────────────
def crea_preprocessore() -> ColumnTransformer:
    """Pipeline di preprocessing: imputazione dei NaN (mediana) + standardizzazione."""
    pipeline_features = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[("features", pipeline_features, FEATURES)],
        remainder="drop",
    )


def _stampa_feature_importance(modello: Pipeline, nome_step: str) -> None:
    """Stampa le 10 feature più importanti del RandomForest dentro la pipeline."""
    rf = modello.named_steps[nome_step]
    imp = pd.Series(rf.feature_importances_, index=FEATURES)
    print("\n=== Top 10 Feature Importance ===")
    print(imp.sort_values(ascending=False).head(10).round(4))


# ─────────────────────────────────────────
# STEP 4 — Classificatore dell'esito (H/D/A)
# ─────────────────────────────────────────
def train_classificatore(df: pd.DataFrame) -> None:
    """Allena, valuta e salva il classificatore dell'esito della partita."""
    print("\n" + "=" * 50)
    print("MODELLO 1 — Classificatore risultato (H/D/A)")
    print("=" * 50)

    X = df[FEATURES]
    y = df[TARGET_CLASS]

    # stratify=y mantiene la stessa proporzione H/D/A in train e test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"✅ Split: {len(X_train)} train | {len(X_test)} test")

    pipeline = Pipeline(steps=[
        ("preprocessor", crea_preprocessore()),
        ("classifier",   RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])

    # Spazio degli iperparametri esplorato dalla ricerca casuale
    param_grid = {
        "classifier__n_estimators":      [100, 200, 300],
        "classifier__max_depth":         [5, 10, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__max_features":      ["sqrt", "log2"],
        "classifier__class_weight":      ["balanced", None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=15,
        scoring="f1_macro",   # f1 macro: tratta le 3 classi alla pari
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    print("\n=== Avvio RandomizedSearch ===")
    search.fit(X_train, y_train)

    print(f"\n✅ Migliori parametri: {search.best_params_}")
    print(f"✅ Miglior F1 macro cv: {search.best_score_:.4f}")

    # Valutazione sul test set
    modello = search.best_estimator_
    y_pred = modello.predict(X_test)
    y_prob = modello.predict_proba(X_test)

    metriche = {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "f1_macro":         float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc":          float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["H", "D", "A"]).tolist(),
        "classes":          list(modello.classes_),
        "best_params":      str(search.best_params_),
        "best_cv_f1":       float(search.best_score_),
    }

    print("\n=== Valutazione test set ===")
    print(f"  Accuracy : {metriche['accuracy']:.4f}")
    print(f"  F1 macro : {metriche['f1_macro']:.4f}")
    print(f"  ROC-AUC  : {metriche['roc_auc']:.4f}")
    print("\n  Confusion Matrix (H/D/A):")
    print(np.array(metriche["confusion_matrix"]))
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    _stampa_feature_importance(modello, "classifier")

    # Salvataggio del modello con metadati e metriche
    MODELS_DIR.mkdir(exist_ok=True)
    package = {
        "pipeline": modello,
        "features": FEATURES,
        "target":   TARGET_CLASS,
        "classes":  list(modello.classes_),
        "metrics":  metriche,
    }
    path = MODELS_DIR / "modello_risultato.joblib"
    joblib.dump(package, path)
    print(f"\n✅ Salvato: {path}")


# ─────────────────────────────────────────
# STEP 5 — Regressore dei gol
# ─────────────────────────────────────────
def train_regressore(df: pd.DataFrame, target: str, nome_file: str) -> None:
    """Allena, valuta e salva un regressore che stima i gol di una squadra."""
    print(f"\n{'=' * 50}")
    print(f"MODELLO — Regressore {target}")
    print("=" * 50)

    X = df[FEATURES]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"✅ Split: {len(X_train)} train | {len(X_test)} test")

    pipeline = Pipeline(steps=[
        ("preprocessor", crea_preprocessore()),
        ("regressor",    RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])

    param_grid = {
        "regressor__n_estimators":      [100, 200, 300],
        "regressor__max_depth":         [5, 10, 20, None],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__max_features":      ["sqrt", "log2"],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=15,
        scoring="neg_mean_absolute_error",   # minimizza l'errore medio sui gol
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    print("\n=== Avvio RandomizedSearch ===")
    search.fit(X_train, y_train)

    print(f"\n✅ Migliori parametri: {search.best_params_}")
    print(f"✅ Miglior MAE cv: {-search.best_score_:.4f}")

    # Valutazione sul test set
    modello = search.best_estimator_
    y_pred = modello.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metriche = {
        "mae":         mae,
        "rmse":        rmse,
        "r2":          r2,
        "best_params": str(search.best_params_),
        "best_cv_mae": float(-search.best_score_),
    }

    print("\n=== Valutazione test set ===")
    print(f"  MAE  : {mae:.4f}  (errore medio gol)")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    _stampa_feature_importance(modello, "regressor")

    # Salvataggio del modello con metadati e metriche
    MODELS_DIR.mkdir(exist_ok=True)
    package = {
        "pipeline": modello,
        "features": FEATURES,
        "target":   target,
        "metrics":  metriche,
    }
    path = MODELS_DIR / nome_file
    joblib.dump(package, path)
    print(f"\n✅ Salvato: {path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main() -> None:
    df = carica_dati()
    analisi_esplorativa(df)

    train_classificatore(df)
    train_regressore(df, TARGET_GOL1, "modello_gol_team1.joblib")
    train_regressore(df, TARGET_GOL2, "modello_gol_team2.joblib")

    print("\n🎯 Training completo!")
    print("  models/modello_risultato.joblib")
    print("  models/modello_gol_team1.joblib")
    print("  models/modello_gol_team2.joblib")


if __name__ == "__main__":
    main()
