# 🏠 Indices de prix immobiliers hédoniques – Normandie

> 🎓 **UE Conduite de Projet** · M1 Statistique & Économétrie · Université de Strasbourg · 2025/26  
> ✍️ Auteurs : **Marius & Nenad**

---

## 🎯 Contexte et objectifs

Le prix immobilier observé dépend à la fois de la **composition** des biens vendus (surface, typologie, neuf/ancien), de l'**évolution du marché** (tendances, cycles, chocs) et de l'**hétérogénéité spatiale** (commune, accessibilité, aménités). Une simple moyenne des prix observés est donc biaisée si la qualité ou la composition des biens vendus varie dans le temps.

Ce projet construit des **indices de prix à qualité constante** par commune et par mois pour les maisons et les appartements, en couvrant les 5 départements normands (14, 27, 50, 61, 76) sur la période 2019–2024.

### 📋 Résultats attendus (cf. slides 4 et 18)

| Livrable | Description |
|---|---|
| 📈 **Indices de prix** | `Index_{c,t}` par commune et par mois, base 100 = juillet 2020 |
| 📉 **Courbes temporelles** | Évolution de quelques communes + benchmarks départementaux |
| 🌡️ **Heatmap commune × temps** | Dynamiques, ruptures et hétérogénéité spatiale |
| 🏆 **Scoring investissement** | Top 3 communes par département selon CAGR, momentum, stabilité |

---

## 📁 Structure du dépôt

```
.
├── conduite_de_projet_marius_nenad.py     # 🐍 Script principal complet
├── Conduite_de_Projet_Marius_Nenad.ipynb  # 📓 Notebook Colab associé
├── data/
│   ├── raw/          # 📦 Fichiers DVF+ téléchargés (.csv.gz)
│   └── processed/    # ✅ dvf_clean.parquet après nettoyage
└── outputs/
    ├── indices/      # 📊 Séries Index_{c,t} au format CSV
    ├── courbes_*.png
    ├── heatmap_*.png
    ├── top3_*.png
    └── tableau_recap_*.csv
```

---

## 🗄️ Données : DVF+

La source de données est le fichier **DVF+** (Demandes de Valeurs Foncières enrichies), disponible en open data sur [data.gouv.fr](https://files.data.gouv.fr/geo-dvf/).

```python
# ── Configuration ──────────────────────────────────────────────────────────────
DEPARTEMENTS = {"14": "Calvados", "27": "Eure", "50": "Manche",
                "61": "Orne",    "76": "Seine-Maritime"}
ANNEES       = list(range(2019, 2025))
BASE_URL     = "https://files.data.gouv.fr/geo-dvf/latest/csv/{annee}/departements/{dep}.csv.gz"

def download_all(force=False):
    for dep in DEPARTEMENTS:
        for annee in ANNEES:
            dest = Path(f"data/raw/dvf_{dep}_{annee}.csv.gz")
            if dest.exists() and not force:
                continue
            r = requests.get(BASE_URL.format(annee=annee, dep=dep), timeout=120)
            if r.status_code == 200:
                dest.write_bytes(r.content)
```

> 💡 Le téléchargement est incrémental : si le fichier existe déjà localement, il n'est pas re-téléchargé (`force=False`).

---

## 🧹 Étape 1 – Nettoyage et construction de $y_i$ (slide 9)

On filtre les **ventes** de maisons et appartements, on calcule le **prix au m²** $p_i$, puis la variable expliquée $y_i = \log(p_i)$. Le logarithme transforme les effets multiplicatifs en effets additifs et stabilise la variance.

```python
# pᵢ = prix au m²,  yᵢ = log(pᵢ)   [slide 9]
df["price_sqm"]     = df["valeur_fonciere"] / df["surface_reelle_bati"]

# 🚫 Filtres outliers
df = df[df["surface_reelle_bati"].between(9, 1_000) &
        df["price_sqm"].between(300, 20_000)]

df["log_price_sqm"] = np.log(df["price_sqm"])   # yᵢ

# 📅 Variables temporelles : t(i) = mois×année, a(i) = année
df["periode"] = (df["annee"].astype(str) + "-"
                 + df["date_mutation"].dt.month.astype(str).str.zfill(2))

# 🏗️ Typologies T (caractéristiques du bien xᵢ)
df["typo_T"] = nb.apply(lambda x: f"T{x}" if 1 <= x <= 4 else ("T5+" if x >= 5 else "T?"))
```

---

## 📐 Étapes 2–4 – Modèle hédonique OLS (slides 10–13)

### 🔢 Le modèle

On estime séparément appartements et maisons le modèle suivant :

$$y_i = \alpha + \mathbf{x}_i'\boldsymbol{\beta} + \gamma_{t(i)} + \delta_{d(i)} + \mu_{c(i)} + \theta_{d(i),a(i)} + \varepsilon_i$$

| Terme | Rôle |
|---|---|
| $\gamma_{t(i)}$ | ⏱️ Effet fixe mois×année → tendance de marché |
| $\mu_{c(i)}$ | 📍 Effet fixe commune → niveau local moyen |
| $\delta_{d(i)}$ | 🗂️ Effet fixe département → contrôle spatial |
| $\theta_{d,a}$ | 📆 Tendances spécifiques département×année |
| $\mathbf{x}_i'\boldsymbol{\beta}$ | 🏠 Caractéristiques du bien (typologies T) |

### ⚙️ Implémentation : dummies et OLS

```python
def estimate_hedonic(df_type, type_bien):
    # 🏘️ Grandes communes (≥ 30 obs.) → dummy μ_c identifiable  [slide 11]
    grande = df["code_commune"].value_counts()
    grande = grande[grande >= MIN_OBS_COMMUNE].index
    df["commune_fe"] = df["code_commune"].where(df["code_commune"].isin(grande), "_petite_")
    df["dep_annee"]  = df["code_dep"] + "_" + df["annee"].astype(str)   # θ_{d,a}

    # Étape 2 – dummies (drop_first = catégorie de référence)  [slide 10]
    X = pd.concat([
        pd.get_dummies(df["periode"],    prefix="t",     drop_first=True, dtype=float),  # γ_t
        pd.get_dummies(df["code_dep"],   prefix="dep",   drop_first=True, dtype=float),  # δ_d
        pd.get_dummies(df["commune_fe"], prefix="com",   drop_first=True, dtype=float),  # μ_c
        pd.get_dummies(df["dep_annee"],  prefix="dep_a", drop_first=True, dtype=float),  # θ_{d,a}
        pd.get_dummies(df["typo_T"],     prefix="typo",  drop_first=True, dtype=float),  # xᵢ
    ], axis=1)

    # 📊 Étape 3 – estimation OLS  [slide 12]
    reg   = LinearRegression(fit_intercept=True, n_jobs=-1).fit(X.values, y)
    y_hat = reg.predict(X.values)
    print(f"  R² = {r2_score(y, y_hat):.4f}")
```

> ⚠️ `drop_first=True` retire une modalité par groupe pour éviter la multicolinéarité parfaite, conformément aux slides 10–11.

### 🧮 Prix net de qualité : $\log P_i^{net}$ (slide 13)

On retire la contribution des variables de qualité pour ne conserver que l'information temporelle et spatiale :

$$\log P_i^{net} = \hat{y}_i - \sum_{k \in \mathcal{H}} \hat{\beta}_k x_{ik}$$

```python
# Étape 4 – log Pᵢⁿᵉᵗ = ŷᵢ − contribution qualité  [slide 13]
qual_idx = [list(X.columns).index(c) for c in X.columns if c.startswith("typo_")]
df["log_price_net"] = y_hat - X.values[:, qual_idx] @ reg.coef_[qual_idx]
```

### 🏘️ Correction des petites communes (slide 15)

Pour les communes sous le seuil (`< 30` transactions), aucun effet fixe $\mu_c$ n'est inclus dans la régression (risque de sur-ajustement). On recale ensuite leur niveau via la moyenne des résidus $\bar{e}_c$ :

$$\log P_i^{net} \leftarrow \log P_i^{net} + \bar{e}_{c(i)}$$

```python
# 🔧 Correction des petites communes  [slide 15]
df["residual"] = y - y_hat
petites = df["commune_fe"] == "_petite_"
if petites.any():
    e_bar = df.loc[petites].groupby("code_commune")["residual"].mean()
    df.loc[petites, "log_price_net"] += (
        df.loc[petites, "code_commune"].map(e_bar).fillna(0).values
    )
```

---

## 📊 Étape 5 – Agrégation et construction de l'indice base 100 (slides 14–17)

### 🔢 Agrégation par (commune, période) — slide 14

On agrège au niveau commune × mois en prenant la **médiane** (robuste aux outliers) :

$$\log P_{c,t}^{net} = \text{médiane}\{\log P_i^{net} : c(i) = c,\ t(i) = t\}$$

```python
agg = (df_type.groupby(["code_commune", "nom_commune", "code_dep", "periode"])
       ["log_price_net"]
       .agg(log_net_med="median", n_obs="count")
       .reset_index())
```

### 🔄 Compléter les périodes manquantes — slide 16

Certaines communes n'ont aucune transaction sur certains mois. On construit d'abord un panel cylindré complet, puis on complète les valeurs manquantes par interpolation linéaire :

```python
# 📐 Panel cylindré {c} × {t} complet
panel = (communes.assign(_k=1)
         .merge(pd.DataFrame({"periode": periodes, "_k": 1}), on="_k")
         .drop("_k", axis=1)
         .merge(agg, on=["code_commune", "nom_commune", "code_dep", "periode"], how="left"))

# 〰️ Interpolation : linéaire → ffill → bfill  [slide 16]
panel["log_net"] = (panel.groupby("code_commune")["log_net_med"]
                         .transform(lambda s: s.interpolate("linear").ffill().bfill()))
```

### 💯 Indice base 100 — slide 17

On fixe la période de référence $t_0$ = **juillet 2020** et on calcule :

$$\text{Index}_{c,t} = 100 \times \exp\!\left(\log P_{c,t}^{net} - \log P_{c,t_0}^{net}\right)$$

```python
REFERENCE_PERIODE = "2020-07"   # t₀

# 📌 Référence par commune
ref = (panel[panel["periode"] == REFERENCE_PERIODE][["code_commune", "log_net"]]
       .rename(columns={"log_net": "log_ref"}))
panel = panel.merge(ref, on="code_commune", how="left")

# ✅ Index_{c,t} = 100 × exp(log P^net_{c,t} − log P^net_{c,t₀})
panel["index_prix"] = 100 * np.exp(panel["log_net"] - panel["log_ref"])
```

---

## 📈 Étape 6 – Visualisations (objectif 2, slide 4)

### 📉 Courbes temporelles

Évolution de $\text{Index}_{c,t}$ pour les principales communes et les benchmarks départementaux :

```python
def plot_temporal(panel, bench, type_bien, top_n=5):
    # 🏙️ Top communes par volume de transactions
    top = panel.groupby("code_commune")["n_obs"].sum().nlargest(top_n).index
    for i, code in enumerate(top):
        s = panel[panel["code_commune"] == code].sort_values("annee_mois")
        ax.plot(s["annee_mois"], s["index_prix"],
                label=f"{s['nom_commune'].iloc[0]} ({s['code_dep'].iloc[0]})")
    ax.axhline(100, color="grey", ls="--", lw=0.8)   # 📍 ligne de référence t₀

    # 🗺️ Benchmarks départementaux
    for dep, label in DEPARTEMENTS.items():
        s = bench[bench["code_dep"] == dep].sort_values("annee_mois")
        ax.plot(s["annee_mois"], s["index_dept"], label=f"{dep} – {label}")
```

### 🌡️ Heatmap commune × temps

Permet de visualiser en un coup d'œil les dynamiques, ruptures et hétérogénéité spatiale (slide 4) :

```python
def plot_heatmap(panel, type_bien, dep="76", top_n=25):
    pivot = (sub[sub["code_commune"].isin(top_c)]
             .pivot_table(index="nom_commune", columns="periode",
                          values="index_prix", aggfunc="first"))
    sns.heatmap(pivot, cmap="RdYlGn", center=100, vmin=70, vmax=150)
```

> 🔴 Rouge = indice < 100 (baisse par rapport à $t_0$) · 🟢 Vert = hausse

---

## 🏆 Étape 7 – Scoring investissement

En s'appuyant sur les indices construits, on calcule pour chaque commune 3 indicateurs puis un score composite, afin d'identifier le **Top 3 communes par département** :

| Indicateur | Formule | Poids |
|---|---|---|
| 📈 CAGR | $(\text{Index}_T / 100)^{1/n} - 1$ | 40 % |
| ⚡ Momentum 12 mois | $\text{Index}_T / \text{Index}_{T-12} - 1$ | 35 % |
| 🛡️ Stabilité | $1 / \text{vol}(\Delta \log \text{Index}_{c,t})$ | 25 % |

```python
def score_communes(panel, type_bien):
    # 📈 CAGR annualisé
    cagr = ((idx_T / 100) ** (1 / n_years) - 1) * 100

    # ⚡ Momentum 12 mois
    momentum = (idx_T / idx_mom - 1) * 100

    # 🛡️ Stabilité = inverse de la volatilité mensuelle
    vol       = np.log(grp["index_prix"]).diff().dropna().std() * 100
    stabilite = 1 / vol if vol > 0 else np.nan

    # 🥇 Score composite : rangs normalisés [0,1] dans le département
    g["score"] = (0.40 * g["rank_cagr_pct"]
                + 0.35 * g["rank_momentum_pct"]
                + 0.25 * g["rank_stabilite"])
```

---

## 🚀 Prérequis et installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn requests pyarrow openpyxl
```

> 📓 Le script peut également être exécuté directement sur **Google Colab** (cf. notebook `.ipynb` joint).

---

## 🔁 Résumé de la pipeline (slide 18)

```
🌐 DVF+ (data.gouv.fr)
         │
         ▼
🧹 ① Nettoyage : yᵢ = log(price_sqm)
         │
         ▼
🎛️ ② Dummies : γ_t, δ_d, μ_c, θ_{d,a}, x_i
         │
         ▼
📐 ③ OLS hédonique (Maisons / Appartements séparément)
         │
         ▼
🧮 ④ log Pᵢⁿᵉᵗ = ŷᵢ − contribution qualité
         │
         ▼
📊 ⑤ Agrégation médiane → log P^net_{c,t}
         │
         ▼
🔄 ⑥ Compléter les dates manquantes (interpolation)
         │
         ▼
💯 ⑦ Index_{c,t} = 100 × exp(log P^net_{c,t} − log P^net_{c,t₀})
         │
         ▼
📈 ⑧ Visualisations + 🏆 Scoring investissement
```
