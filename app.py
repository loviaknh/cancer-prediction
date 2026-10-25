# -*- coding: utf-8 -*-
"""
EEIA 2026 — Interface de pronostic (cancer du sein, MSK-IMPACT 2025)  — v3
Cox PH (11 var.) + AFT Weibull (4 var.) — pipeline identique au notebook.

Lancement :  streamlit run app.py
Fichiers :   models/cox_model.pkl  (requis)
             models/scaler.pkl     (requis pour des prédictions fiables)
             models/aft_model.pkl  (optionnel — active le volet AFT)
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Constantes du pipeline (notebook de référence) ----------
FGA_HIGH_THRESHOLD = 0.2
MSI_SCORE_IMPUT_MEDIAN = 0.430
TUMOR_PURITY_MEDIAN = 40.0
FEATURES_REDUCED = ["mutation_count", "is_metastatic", "msi_score", "fga_high"]
CINDEX = {"Cox PH": (0.671, 0.661), "AFT Weibull": (0.670, 0.661)}  # (train, test)

MODEL_DIR = Path("models")

# ---------- Chargement ----------
@st.cache_resource
def load_artifacts():
    cox = joblib.load(MODEL_DIR / "cox_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl") if (MODEL_DIR / "scaler.pkl").exists() else None
    aft_p = MODEL_DIR / "aft_model.pkl"
    aft = joblib.load(aft_p) if aft_p.exists() else None
    return cox, scaler, aft

def make_features(sample_type, fga, mutation_count, msi_score, msi_type,
                  tumor_purity, sample_coverage, scaler):
    """Reproduit exactement le pipeline d'entraînement. Retourne (X_cox, X_aft)."""
    is_meta = int(sample_type == "Métastatique")
    fga_high = int(fga >= FGA_HIGH_THRESHOLD)
    row = {
        "fga_high": fga_high,
        "is_metastatic": is_meta,
        "meta_x_fga": is_meta * fga_high,
        "msi_score": float(msi_score),
        "msi_type_do not report": int(msi_type == "do not report"),
        "msi_type_indeterminate": int(msi_type == "indeterminate"),
        "msi_type_instable": int(msi_type == "instable"),
        "msi_type_stable": int(msi_type == "stable"),
        "mutation_count": np.log1p(float(mutation_count)),
        "sample_coverage": float(sample_coverage),
        "tumor_purity": float(tumor_purity),
    }
    X = pd.DataFrame([row])
    if scaler is not None:
        cols = list(getattr(scaler, "feature_names_in_",
                            ["msi_score", "mutation_count", "tumor_purity", "sample_coverage"]))
        X[cols] = scaler.transform(X[cols])
    return X, X[FEATURES_REDUCED]

def s_at(sf, t):
    return float(sf.iloc[np.argmin(np.abs(sf.index.values - t)), 0])

def fmt_months(v):
    return f"{v:.0f} mois" if np.isfinite(v) else "> 132 mois"

# ---------- Page ----------
st.set_page_config(page_title="Pronostic cancer du sein — EEIA", page_icon="🎗️",
                   layout="wide")
st.title("🎗️ Analyse de survie — Cancer du sein (MSK-IMPACT 2025)")
st.caption("Projet pédagogique EEIA Bénin. **Ne constitue en aucun cas un outil de "
           "décision clinique.**")

cox, scaler, aft = load_artifacts()
if scaler is None:
    st.error("`models/scaler.pkl` manquant — prédictions NON fiables.")

# ---------- Saisie (sidebar, partagée) ----------
def patient_inputs(prefix="", defaults=None):
    d = defaults or {}
    sample_type = st.radio("Type d'échantillon",
                           ["Primaire", "Métastatique", "Récidive locale"],
                           index=d.get("st", 0), key=prefix + "st")
    fga = st.slider("Fraction Genome Altered", 0.0, 1.0, d.get("fga", 0.15), 0.01,
                    key=prefix + "fga")
    st.caption(f"→ fga_high = {int(fga >= FGA_HIGH_THRESHOLD)} (seuil {FGA_HIGH_THRESHOLD})")
    mut = st.number_input("Mutation Count", 0, value=d.get("mut", 5), step=1,
                          key=prefix + "mut")
    msi_known = st.checkbox("MSI Score connu", True, key=prefix + "msik")
    msi = st.number_input("MSI Score", value=d.get("msi", 0.43), step=0.1,
                          format="%.2f", disabled=not msi_known, key=prefix + "msi")
    if not msi_known:
        msi = MSI_SCORE_IMPUT_MEDIAN
    msi_type = st.selectbox("MSI Type", ["stable", "instable", "indeterminate",
                                         "do not report"],
                            index=d.get("mt", 0), key=prefix + "mt")
    purity = st.slider("Tumor Purity (%)", 0, 100, d.get("pur", int(TUMOR_PURITY_MEDIAN)),
                       key=prefix + "pur")
    cov = st.number_input("Sample Coverage", 0, value=d.get("cov", 600), step=10,
                          key=prefix + "cov")
    return dict(sample_type=sample_type, fga=fga, mutation_count=mut, msi_score=msi,
                msi_type=msi_type, tumor_purity=purity, sample_coverage=cov)

with st.sidebar:
    st.header("👤 Patient")
    p = patient_inputs()

X_cox, X_aft = make_features(scaler=scaler, **p)

tab_pred, tab_ab, tab_types, tab_pedag = st.tabs(
    ["🔮 Prédiction", "⚖️ Comparer 2 profils", "🧬 Profils types",
     "🎓 Modèles & pédagogie"])

# ================= Onglet 1 : Prédiction =================
with tab_pred:
    ph = float(cox.predict_partial_hazard(X_cox).iloc[0])
    sf = cox.predict_survival_function(X_cox)
    med_cox = float(np.asarray(cox.predict_median(X_cox)).ravel()[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Risque relatif (Cox)", f"{ph:.2f}×",
              delta=f"{'+' if ph >= 1 else '−'}{abs(ph - 1) * 100:.0f}% vs profil moyen",
              delta_color="inverse")
    c2.metric("Survie médiane (Cox)", fmt_months(med_cox))
    if aft is not None:
        q50 = float(aft.predict_percentile(X_aft, p=0.5).iloc[0])
        q25 = float(aft.predict_percentile(X_aft, p=0.75).iloc[0])   # convention notebook
        q75 = float(aft.predict_percentile(X_aft, p=0.25).iloc[0])
        c3.metric("Survie médiane (AFT Weibull)", fmt_months(q50),
                  delta=f"IQR : {q25:.0f} – {fmt_months(q75)}", delta_color="off")
    else:
        c3.metric("AFT Weibull", "non chargé")
        c3.caption("Placer `aft_model.pkl` dans `models/`.")

    # ---- Position du patient sur l'échelle de risque ----
    st.markdown("##### Où se situe ce profil ?")
    fig, ax = plt.subplots(figsize=(9, 0.9))
    ax.axvspan(-1.5, -0.3, color="#2e7d32", alpha=0.25)
    ax.axvspan(-0.3, 0.3, color="#f9a825", alpha=0.25)
    ax.axvspan(0.3, 1.5, color="#c62828", alpha=0.25)
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.scatter([np.clip(np.log(ph), -1.45, 1.45)], [0], s=250, marker="v",
               color="#1565c0", zorder=5)
    ax.set_xlim(-1.5, 1.5); ax.set_yticks([])
    ax.set_xticks([-1.5, 0, 1.5])
    ax.set_xticklabels(["risque plus faible", "profil moyen cohorte", "risque plus élevé"])
    for s in ax.spines.values(): s.set_visible(False)
    st.pyplot(fig)

    # ---- Courbe patient vs cohorte moyenne ----
    st.markdown("##### Courbe de survie : ce profil vs profil moyen de la cohorte")
    base = cox.baseline_survival_
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(base.index, base.iloc[:, 0], color="gray", lw=1.5, ls="--",
            label="Profil moyen (baseline)")
    ax.plot(sf.index, sf.iloc[:, 0], color="#1565c0", lw=2.4, label="Ce profil (Cox)")
    if aft is not None:
        sf_aft = aft.predict_survival_function(X_aft)
        ax.plot(sf_aft.index, sf_aft.iloc[:, 0], color="#e65100", lw=1.8, ls=":",
                label="Ce profil (AFT Weibull)")
    for t in (12, 36, 60):
        ax.scatter([t], [s_at(sf, t)], zorder=5, color="#1565c0")
        ax.annotate(f"{s_at(sf, t):.0%}", (t, s_at(sf, t)),
                    textcoords="offset points", xytext=(6, 7), fontsize=9)
    ax.set_xlabel("Temps (mois)"); ax.set_ylabel("S(t)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    st.pyplot(fig)

    # ---- Export ----
    resume = pd.DataFrame([{**p, "hazard_partiel_cox": round(ph, 3),
                            "surv_mediane_cox_mois": round(med_cox, 1) if np.isfinite(med_cox) else ">132",
                            "S12": round(s_at(sf, 12), 3), "S36": round(s_at(sf, 36), 3),
                            "S60": round(s_at(sf, 60), 3)}])
    st.download_button("💾 Exporter cette prédiction (CSV)",
                       resume.to_csv(index=False).encode(),
                       "prediction_patient.csv", "text/csv")

# ================= Onglet 2 : Comparaison A/B =================
with tab_ab:
    st.markdown("Comparer deux profils côte à côte — idéal pour illustrer l'effet "
                "d'une variable *toutes choses égales par ailleurs*.")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### 🅰️ Profil A")
        pA = patient_inputs("A_", defaults={"st": 0, "fga": 0.15})
    with colB:
        st.markdown("#### 🅱️ Profil B")
        pB = patient_inputs("B_", defaults={"st": 1, "fga": 0.45})

    XA, XAa = make_features(scaler=scaler, **pA)
    XB, XBa = make_features(scaler=scaler, **pB)
    sfA, sfB = cox.predict_survival_function(XA), cox.predict_survival_function(XB)
    phA = float(cox.predict_partial_hazard(XA).iloc[0])
    phB = float(cox.predict_partial_hazard(XB).iloc[0])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(sfA.index, sfA.iloc[:, 0], lw=2.4, color="#1565c0", label=f"A (hazard {phA:.2f})")
    ax.plot(sfB.index, sfB.iloc[:, 0], lw=2.4, color="#c62828", label=f"B (hazard {phB:.2f})")
    ax.set_xlabel("Temps (mois)"); ax.set_ylabel("S(t)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    st.pyplot(fig)

    rows = []
    for name, X_, sf_ in [("A", XA, sfA), ("B", XB, sfB)]:
        med = float(np.asarray(cox.predict_median(X_)).ravel()[0])
        rows.append({"Profil": name,
                     "Hazard partiel": round(float(cox.predict_partial_hazard(X_).iloc[0]), 2),
                     "S(12 m)": f"{s_at(sf_, 12):.0%}", "S(36 m)": f"{s_at(sf_, 36):.0%}",
                     "S(60 m)": f"{s_at(sf_, 60):.0%}", "Médiane": fmt_months(med)})
    st.dataframe(pd.DataFrame(rows).set_index("Profil"), use_container_width=True)
    st.info(f"Risque instantané du profil B : **{phB/phA:.2f}×** celui du profil A.")

# ================= Onglet 3 : Profils types (AFT) =================
with tab_types:
    st.markdown("Les 4 profils types du notebook (autres variables aux valeurs "
                "médianes du train), en quantiles de survie prédits par l'AFT — "
                "la formulation la plus directement communicable.")
    if aft is None:
        st.warning("`aft_model.pkl` absent — placer le fichier dans `models/` "
                    "pour activer cet onglet.")
    else:
        base_row, _ = make_features("Primaire", 0.15, 5, MSI_SCORE_IMPUT_MEDIAN,
                                    "stable", TUMOR_PURITY_MEDIAN, 600, scaler)
        profs, names = [], ["Primaire / FGA bas", "Primaire / FGA haut",
                            "Méta / FGA bas", "Méta / FGA haut"]
        for meta, fgah in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            r = base_row.copy()
            r["is_metastatic"], r["fga_high"] = meta, fgah
            profs.append(r[FEATURES_REDUCED])
        P = pd.concat(profs); P.index = names
        out = pd.DataFrame(index=names)
        for label, pq in [("Q25 (mois)", 0.75), ("Médiane (mois)", 0.5), ("Q75 (mois)", 0.25)]:
            out[label] = aft.predict_percentile(P, p=pq).values
        st.dataframe(out.round(1), use_container_width=True)
        st.caption("Q25 : 25 % des patients de ce profil ont eu l'événement avant "
                   "cette durée ; Q75 : 75 %. L'écart Q25–Q75 illustre l'incertitude "
                   "individuelle — à souligner en séance.")

# ================= Onglet 4 : Modèles & pédagogie =================
with tab_pedag:
    st.markdown("#### Hazard Ratios du modèle de Cox (avec IC 95 %)")
    summ = cox.summary
    fig, ax = plt.subplots(figsize=(8, 5))
    order = summ["coef"].sort_values().index
    y = np.arange(len(order))
    ax.errorbar(np.exp(summ.loc[order, "coef"]), y,
                xerr=[np.exp(summ.loc[order, "coef"]) - np.exp(summ.loc[order, "coef lower 95%"]),
                      np.exp(summ.loc[order, "coef upper 95%"]) - np.exp(summ.loc[order, "coef"])],
                fmt="o", color="#1565c0", ecolor="#90a4ae", capsize=3)
    ax.axvline(1, color="gray", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(order)
    ax.set_xlabel("Hazard Ratio (échelle log)"); ax.set_xscale("log")
    ax.grid(alpha=0.3, axis="x")
    st.pyplot(fig)
    st.caption("⚠️ Variables continues standardisées : le HR de `mutation_count` "
               "s'interprète **par écart-type de log(1+mutations)**, pas par mutation. "
               "HR > 1 → risque accru ; IC croisant 1 → effet non significatif.")

    st.markdown("#### Comparaison des modèles (C-index)")
    comp = pd.DataFrame([{"Modèle": k, "C-index train": v[0], "C-index test": v[1],
                          "Écart (sur-apprentissage)": round(v[0] - v[1], 3)}
                         for k, v in CINDEX.items()])
    st.dataframe(comp.set_index("Modèle"), use_container_width=True)
    st.caption("0,5 = hasard ; 0,65–0,75 = déjà informatif en clinique. "
               "Le RSF (1,5 Go) est volontairement exclu de l'interface — voir notebook.")

    st.markdown("#### Limites et biais à discuter")
    st.markdown("""
- **Représentativité** : données du MSKCC (New York) — biais de sélection vis-à-vis
  des populations africaines (accès au NGS, sous-types moléculaires, stade au diagnostic).
- **Hypothèse PH** partiellement violée dans le modèle complet → modèles stratifié
  et réduit dans le notebook.
- **AFT paramétrique** : suppose une distribution Weibull (choisie par AIC, pas décrétée).
- Prédictions valides uniquement dans le domaine des données d'entraînement.
""")
