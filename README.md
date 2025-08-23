# app.py — modèle embarqué (chargé automatiquement depuis le dossier de l'app)
import io, math
import numpy as np, pandas as pd, streamlit as st, joblib
from pathlib import Path

# =================== Config & constantes ===================
st.set_page_config(page_title="Prédiction des classes à construire", layout="centered")

CLASSES_PER_SCHOOL = 6
DEFAULT_FEATURES = [
    "pop_scolarisable","taux_croissance_pct","gratuité","situation_secu",
    "eleves_inscrits","taux_accroiss_eleves_pct","taux_scolarisation_pct","nb_ecoles"
]
DEFAULT_RENAME_MAP = {
    "Population Scolarisable potientiel au primaire": "pop_scolarisable",
    "Taux de croissance anunuel %": "taux_croissance_pct",
    "Gratuité d'Enseignement": "gratuité",
    "Situation sécuritaire": "situation_secu",
    "Élèves inscrits au primaire": "eleves_inscrits",
    "Taux d'Accroissement des Élèves au primaire en %": "taux_accroiss_eleves_pct",
    "Taux de scolarisation en %": "taux_scolarisation_pct",
    "Nombre d'Ecoles  Primaires": "nb_ecoles",
}

# Chemin du modèle .pkl (placé à côté de app.py)
MODEL_PATH = Path(__file__).parent / "modele_nb_classes.pkl"

# =================== Chargement du modèle ===================
st.title("🏫 Prédiction du nombre de classes à construire")
st.caption("Pipeline scikit-learn embarqué — conversion classes → écoles (6 classes/école)")

@st.cache_resource(show_spinner=True)
def load_embedded_model(path: Path):
    bundle = joblib.load(path)
    # Supporte .pkl sauvé comme dict {pipeline, features, rename_map, …} ou pipeline seul
    if isinstance(bundle, dict):
        pipeline = bundle["pipeline"]
        expected_features = bundle.get("features", None)
        rename_map = bundle.get("rename_map", DEFAULT_RENAME_MAP)
    else:
        pipeline = bundle
        expected_features, rename_map = None, DEFAULT_RENAME_MAP

    # Si les features ne sont pas stockées, on tente de les déduire du ColumnTransformer
    if expected_features is None:
        try:
            expected_features = list(pipeline.named_steps["prep"].transformers_[0][2])
        except Exception:
            expected_features = DEFAULT_FEATURES
    return pipeline, expected_features, rename_map

if not MODEL_PATH.exists():
    st.error(f"Le fichier modèle est introuvable : {MODEL_PATH}\n"
             f"➡️ Placez **modele_nb_classes.pkl** dans le même dossier que **app.py** puis relancez.")
    st.stop()

try:
    pipeline, expected_features, rename_map = load_embedded_model(MODEL_PATH)
    st.success("Modèle chargé automatiquement ✅")
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    st.stop()

# =================== Saisie des variables ===================
st.header("🧮 Variables explicatives & année")
colA, colB = st.columns(2)
with colA:
    target_year = st.number_input("Année cible", 2024, 2100, 2030, 1)
with colB:
    existing_classes = st.number_input(
        "Classes existantes (optionnel)", min_value=0, value=0, step=1,
        help="Si renseigné, on calcule les classes/écoles à construire en plus."
    )

st.markdown("#### Valeurs attendues **pour l’année cible** (ex. 2030)")
c1, c2 = st.columns(2)
with c1:
    pop_scolarisable = st.number_input("Population scolarisable (primaire)", min_value=0.0, value=1.40e7, step=1e5, format="%.0f")
    taux_croissance_pct = st.number_input("Taux de croissance annuel (%)", -100.0, 100.0, 3.0, 0.1)
    gratuité = st.selectbox("Gratuité d’enseignement (0/1)", options=[0, 1], index=1)
    situation_secu = st.selectbox("Situation sécuritaire (0/1)", options=[0, 1], index=1)
with c2:
    eleves_inscrits = st.number_input("Élèves inscrits au primaire", min_value=0.0, value=9.8e6, step=1e5, format="%.0f")
    taux_accroiss_eleves_pct = st.number_input("Taux d’accroissement des élèves (%)", -100.0, 100.0, 2.0, 0.1)
    taux_scolarisation_pct = st.number_input("Taux de scolarisation (%)", 0.0, 100.0, 70.0, 0.1)
    nb_ecoles = st.number_input("Nombre d’écoles primaires", min_value=0, value=32000, step=100)

def build_features_df(features: list, rename_map: dict) -> pd.DataFrame:
    row = {
        "pop_scolarisable": pop_scolarisable,
        "taux_croissance_pct": taux_croissance_pct,
        "gratuité": float(gratuité),
        "situation_secu": float(situation_secu),
        "eleves_inscrits": eleves_inscrits,
        "taux_accroiss_eleves_pct": taux_accroiss_eleves_pct,
        "taux_scolarisation_pct": taux_scolarisation_pct,
        "nb_ecoles": float(nb_ecoles),
    }
    df_in = pd.DataFrame([row]).rename(columns=rename_map or {})
    # Ajout des colonnes manquantes + tri dans l'ordre attendu par le pipeline
    for c in features:
        if c not in df_in.columns:
            df_in[c] = np.nan
    return df_in[features]

# =================== Prédiction ===================
if st.button("🔮 Prédire le nombre de classes"):
    X_new = build_features_df(expected_features, rename_map)
    with st.expander("Données envoyées au modèle", expanded=False):
        st.dataframe(X_new)

    try:
        y_pred = pipeline.predict(X_new)
        pred_classes = int(max(round(float(y_pred[0])), 0))

        st.subheader(f"Résultat pour {target_year}")
        st.metric("Nombre de classes (prédites)", f"{pred_classes:,}".replace(",", " "))

        to_build_classes = max(pred_classes - int(existing_classes), 0)
        st.metric("Classes à construire (après stock existant)", f"{to_build_classes:,}".replace(",", " "))

        schools_needed = (to_build_classes + CLASSES_PER_SCHOOL - 1)//CLASSES_PER_SCHOOL if to_build_classes > 0 else 0
        st.metric(f"Écoles à construire (≈ {CLASSES_PER_SCHOOL} classes/école)", f"{schools_needed:,}".replace(",", " "))

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")

st.info("💡 Pour une projection 2030 réaliste, saisissez des valeurs **déjà projetées** (population 2030, taux attendus, etc.).")
