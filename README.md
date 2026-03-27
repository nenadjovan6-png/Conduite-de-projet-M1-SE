# 🏠 Indices de Prix Immobiliers Hédoniques – Normandie

> **UE Conduite de Projet** · Master 1 Statistique & Économétrie  
> Université de Strasbourg · 2025/26  
> Auteurs : **Marius & Nenad**

---

## 📋 Présentation du projet

Ce projet met en œuvre la méthodologie des **indices de prix hédoniques** présentée dans le cours *UE Conduite de Projet* pour construire des indices de prix immobiliers par **commune** et par **mois** sur la région Normandie (2019–2024).

### Problème posé (Slide 3 du cours)

Le prix immobilier observé d'une période à l'autre ne reflète pas seulement l'évolution du marché : il dépend aussi de la **composition** des biens vendus (surface, typologie) et de l'**hétérogénéité spatiale** (commune, département). Comparer directement les prix moyens bruts dans le temps est donc trompeur.

### Solution : l'indice hédonique à qualité constante

On estime un modèle de régression qui **neutralise l'effet des caractéristiques des biens** pour ne conserver que le signal pur de marché. On en déduit un indice $\text{Index}_{c,t}$ (base 100) pour chaque commune $c$ et chaque période $t$.

### Résultats produits (Slide 4 du cours)

| Livrable | Description |
|---|---|
| `courbes_*.png` | Courbes temporelles de $\text{Index}_{c,t}$ – top communes + benchmarks départementaux |
| `heatmap_*_dep**.png` | Heatmap commune × temps par département |
| `carte_choroplethe_*_croissance.png` | Carte choroplèthe – croissance des prix depuis $t_0$ |
| `carte_choroplethe_*_index_fin.png` | Carte choroplèthe – niveau absolu de l'indice |
| `tableau_recap_*.csv` | Croissance totale, CAGR, nombre de transactions par département |
| `top3_*.png` / `top3_*.csv` | Top 3 communes par département selon un score composite d'investissement |
| `scoring_*.csv` | Score complet de toutes les communes |

---
---

## ⚙️ Prérequis & Installation

Le projet tourne sur **Google Colab** (recommandé) ou en local avec Python 3.10+.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn requests pyarrow openpyxl geopandas
```

Ouvrir le notebook sur Colab :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oZ-EvxbsCP1PyDeLL9e5Q9jP4KLSj8xT)

---

## 📐 Méthodologie détaillée & explication des blocs de code

Le README suit fidèlement le **résumé opérationnel du cours (Slide 18)** en 7 étapes numérotées.

---

### Bloc 0 — Dépendances & arborescence

```python
!pip install -q numpy pandas scikit-learn matplotlib seaborn requests pyarrow openpyxl
for d in ["data/raw", "data/processed", "outputs"]:
    Path(d).mkdir(parents=True, exist_ok=True)
```

Mise en place de l'environnement. Les dossiers `data/` et `outputs/` sont créés automatiquement pour organiser les données brutes, les données traitées et les sorties graphiques.

---

### Bloc 1 — Configuration

```python
DEPARTEMENTS      = {"14":"Calvados", "27":"Eure", ...}
ANNEES            = list(range(2019, 2025))
REFERENCE_PERIODE = "2020-07"    # t₀ – base 100
MIN_OBS_COMMUNE   = 30           # seuil d'identification de μ_c
PRICE_SQM_MIN, PRICE_SQM_MAX = 300, 20_000
SURFACE_MIN, SURFACE_MAX     = 9, 1_000
```

Centralisation de tous les paramètres du projet. Deux paramètres sont directement issus du cours :

- `REFERENCE_PERIODE` : la **période de référence $t_0$** (Slide 17) à laquelle $\text{Index}_{c,t_0} = 100$ pour toutes les communes.
- `MIN_OBS_COMMUNE` : le **seuil d'identification de l'effet fixe commune** $\mu_c$ (Slide 11). En dessous de 30 transactions, la dummy commune est mal identifiée (sur-ajustement) et n'est pas incluse dans le modèle.

---

### Bloc 2 — Téléchargement des données DVF+

```python
def download_all(force=False):
    r = requests.get(BASE_URL.format(annee=annee, dep=dep), timeout=120)
    dest.write_bytes(r.content)
```

**Source de données (Slide 6 du cours)** : DVF+ (*Demandes de Valeurs Foncières*), base open-data du gouvernement français. Elle contient toutes les transactions immobilières enregistrées aux hypothèques, avec le prix, la surface, le type de bien et la localisation. On télécharge 30 fichiers (5 départements × 6 années), compressés en `.csv.gz`.

---

### Bloc 3 — Chargement et nettoyage — *Étape 1 du cours*

> **Cours Slide 9 :** *"Nettoyer les transactions et construire $y_i = \log(\text{price\_sqm}_i)$"*

```python
df["price_sqm"]     = df["valeur_fonciere"] / df["surface_reelle_bati"]   # pᵢ
df["log_price_sqm"] = np.log(df["price_sqm"])                             # yᵢ = log(pᵢ)
```

Cette étape construit la **variable expliquée** $y_i = \log(p_i)$ du modèle hédonique.

**Pourquoi le logarithme ? (Slide 9)** Le log transforme les effets multiplicatifs en effets additifs et stabilise la variance des prix, qui sont très asymétriques à droite. Un coefficient $\hat{\beta}_k$ s'interprète directement comme un effet semi-élasticité : une unité supplémentaire de la caractéristique $k$ augmente le prix de $100 \times \hat{\beta}_k$%.

Les filtres appliqués :
- Uniquement les **ventes** (`nature_mutation == "Vente"`) — pas les successions ni les donations
- Uniquement **Maisons et Appartements** (`type_local`)
- Exclusion des biens aberrants : surfaces hors $[9, 1000]$ m² et prix au m² hors $[300, 20\,000]$ €

On construit aussi les **variables structurelles** du modèle (Slide 9) :
- `periode` = mois × année → indexe le temps $t(i)$
- `code_dep`, `code_commune` → indexent la localisation $d(i)$ et $c(i)$
- `typo_T` (T1 à T5+) → les caractéristiques du bien $x_i$

---

### Bloc 4 — Modèle hédonique OLS — *Étapes 2, 3 et 4 du cours*

C'est le cœur du projet. On estime séparément un modèle pour les **Maisons** et un pour les **Appartements** (Slide 10).

#### Le modèle (Slides 10–11)

$$y_i = \alpha + x_i'\beta + \gamma_{t(i)} + \delta_{d(i)} + \mu_{c(i)} + \theta_{d(i),a(i)} + \varepsilon_i$$

| Terme | Rôle dans le code | Slides |
|---|---|---|
| $x_i'\beta$ | dummies `typo_` — caractéristiques du bien (T1–T5+) | 9, 10 |
| $\gamma_{t(i)}$ | dummies `t_` — effet fixe mois × année = tendance de marché | 10 |
| $\delta_{d(i)}$ | dummies `dep_` — effet fixe département = contrôle spatial | 10 |
| $\mu_{c(i)}$ | dummies `com_` — effet fixe commune = niveau local moyen | 10, 11 |
| $\theta_{d(i),a(i)}$ | dummies `dep_a_` — tendances spécifiques département × année | 10 |

```python
X = pd.concat([
    pd.get_dummies(df["periode"],    prefix="t",     drop_first=True),  # γ_t
    pd.get_dummies(df["code_dep"],   prefix="dep",   drop_first=True),  # δ_d
    pd.get_dummies(df["commune_fe"], prefix="com",   drop_first=True),  # μ_c
    pd.get_dummies(df["dep_annee"],  prefix="dep_a", drop_first=True),  # θ_{d,a}
    pd.get_dummies(df["typo_T"],     prefix="typo",  drop_first=True),  # xᵢ
], axis=1)
```

> **`drop_first=True`** correspond au principe du cours (Slide 10) : *"on retire une modalité par groupe (catégorie de référence) pour éviter la colinéarité."*

#### Traitement des petites communes (Slide 11)

```python
grande = df["code_commune"].value_counts()
grande = grande[grande >= MIN_OBS_COMMUNE].index
df["commune_fe"] = df["code_commune"].where(df["code_commune"].isin(grande), "_petite_")
```

Pour les communes avec moins de 30 transactions, aucune dummy $\mu_c$ n'est créée. Toutes ces communes sont regroupées sous `_petite_`. Le cours justifie : *"avec un faible effectif, une dummy commune est mal identifiée (forte variance, risques de sur-ajustement)."*

#### Estimation OLS (Slide 12)

```python
reg   = LinearRegression(fit_intercept=True, n_jobs=-1).fit(Xv, y)
y_hat = reg.predict(Xv)
print(f"  R² = {r2_score(y, y_hat):.4f}")
```

$$\hat{\theta} = \arg\min_\theta \|y - X\theta\|_2^2$$

Le $R^2$ est affiché comme **diagnostic de qualité du modèle** (Slide 12).

#### Prix net de qualité — log $P_i^{\text{net}}$ (Slide 13)

```python
qual_idx = [list(X.columns).index(c) for c in X.columns if c.startswith("typo_")]
df["log_price_net"] = y_hat - Xv[:, qual_idx] @ reg.coef_[qual_idx]
```

$$\log P_i^{\text{net}} = \hat{y}_i - \sum_{k \in \mathcal{H}} \hat{\beta}_k x_{ik}$$

On soustrait la contribution des variables de qualité $\mathcal{H}$ (typologies T). $\hat{y}_i$ contient *temps + lieu + qualité* ; $\log P_i^{\text{net}}$ ne conserve que *temps + lieu*. Cela permet de mesurer une évolution **à qualité constante**, indépendamment du mix de biens vendus.

#### Correction des petites communes (Slide 15)

```python
e_bar = df.loc[petites].groupby("code_commune")["residual"].mean()
df.loc[petites, "log_price_net"] += df.loc[petites, "code_commune"].map(e_bar).fillna(0).values
```

$$\log P_i^{\text{net}} \leftarrow \log P_i^{\text{net}} + \bar{e}_{c(i)}$$

Pour les communes sans effet fixe, on recale le niveau en ajoutant la **moyenne des résidus par commune** $\bar{e}_c$, comme préconisé en Slide 15.

---

### Bloc 5 — Construction de l'indice base 100 — *Étapes 5 et 6 du cours*

#### Agrégation par commune × période (Slide 14)

```python
agg = df_type.groupby(["code_commune","nom_commune","code_dep","periode"])["log_price_net"]
             .agg(log_net_med="median", n_obs="count")
```

$$\log P_{c,t}^{\text{net}} = \text{médiane}\{ \log P_i^{\text{net}} : c(i)=c,\; t(i)=t \}$$

La **médiane** est recommandée par le cours (Slide 14) pour sa robustesse aux valeurs aberrantes résiduelles.

#### Panel cylindré + complétion des dates manquantes (Slide 16)

```python
panel["log_net"] = (panel.groupby("code_commune")["log_net_med"]
                         .transform(lambda s: s.interpolate("linear").ffill().bfill()))
```

On construit d'abord la **grille complète** $\{(c,t)\}$ pour toutes les communes et toutes les périodes. Les cellules vides (aucune transaction cette période dans cette commune) sont remplies par interpolation linéaire, puis forward/backward fill — garantissant **zéro NaN** dans l'indice final *pour les communes qui ont au moins une transaction sur l'ensemble de la période*.

#### Calcul de l'indice (Slide 17)

```python
panel["index_prix"] = 100 * np.exp(panel["log_net"] - panel["log_ref"])
```

$$\text{Index}_{c,t} = 100 \times \exp\!\left(\log P_{c,t}^{\text{net}} - \log P_{c,t_0}^{\text{net}}\right)$$

Si $\text{Index}_{c,t} = 120$, les prix ont augmenté de **+20 %** dans la commune $c$ depuis la période de référence $t_0$, à qualité constante.

Un **benchmark départemental** est également calculé (médiane de $\log P_{c,t}^{\text{net}}$ sur toutes les communes du département), utile pour la comparaison interdépartementale.

---

### Bloc 6 — Visualisations — *Objectif (2) du cours (Slide 4)*

#### 6a — Courbes temporelles

Deux panneaux côte à côte :
- **Gauche** : évolution de $\text{Index}_{c,t}$ pour le top 5 des communes par volume de transactions
- **Droite** : benchmarks départementaux pour les 5 départements normands

#### 6b — Heatmap commune × temps

```python
sns.heatmap(pivot, cmap="RdYlGn", center=100, vmin=70, vmax=150)
```

Le cours (Slide 4) décrit cet outil comme permettant de visualiser *"les dynamiques, ruptures et hétérogénéité spatiale"*. La palette `RdYlGn` est centrée sur 100 (rouge = baisse, vert = hausse). Une heatmap est produite par département.

#### 6c — Tableau récapitulatif départemental

Calcule pour chaque département :

| Indicateur | Formule |
|---|---|
| $\text{Index}_{t_{\max}}$ | valeur finale de l'indice départemental |
| Croissance totale | $(\text{Index}_{t_{\max}} / \text{Index}_{t_{\min}} - 1) \times 100$ |
| CAGR | $(\text{Index}_{t_{\max}} / 100)^{1/n} - 1$ |

#### 6d — Carte choroplèthe (Slide 4)

Le cours demande explicitement une *"carte choroplèthe : niveau ou croissance de l'indice à une date donnée"*. Deux variantes sont produites pour chaque type de bien :

- **Croissance** : $\text{Index}_{c,t_{\max}} - 100$, soit la variation en % depuis $t_0$
- **Niveau absolu** : la valeur de $\text{Index}_{c,t_{\max}}$

La palette `RdYlGn` est centrée sur 0 (rouge = baisse de prix, vert = hausse). Les géométries des communes sont téléchargées en temps réel depuis l'API officielle `geo.api.gouv.fr`, sans fichier shapefile à fournir.

---

### Bloc 7 — Analyse d'investissement : Top 3 communes par département

Ce bloc prolonge le projet en exploitant $\text{Index}_{c,t}$ (Slides 17–18) pour **classer les communes selon leur attractivité d'investissement**.

#### Score composite

Pour chaque commune, trois indicateurs sont calculés à partir de la série $\text{Index}_{c,t}$ :

| Indicateur | Formule | Poids dans le score |
|---|---|---|
| **CAGR** | $(\text{Index}_T / 100)^{1/n} - 1$ | 40 % |
| **Momentum 12 mois** | $(\text{Index}_T / \text{Index}_{T-12}) - 1$ | 35 % |
| **Stabilité** | $1 / \sigma(\Delta \log \text{Index}_{c,t})$ | 25 % |

```python
g["score"] = (0.40 * g["rank_cagr_pct"]
            + 0.35 * g["rank_momentum_pct"]
            + 0.25 * g["rank_stabilite"])
```

Les rangs sont normalisés dans $[0,1]$ **au sein de chaque département** pour rendre les scores comparables. Le Top 3 par département est visualisé avec la courbe $\text{Index}_{c,t}$ annotée du CAGR et du Momentum.

---
### Bloc 8 — Évolution régionale : Maisons vs Appartements — *Objectif (2) du cours (Slide 4)*

Ce bloc produit une comparaison directe de l'évolution des prix à l'échelle de toute la Normandie entre les deux types de biens.

#### Benchmark régional
```python
reg = panel.groupby("periode")["log_price_net"].median().reset_index()
reg["index_regional"] = 100 * np.exp(reg["log_net_reg"] - log_ref)
```

On agrège les `log_price_net` de toutes les communes et tous les départements par période via la **médiane régionale** (Slide 14), puis on applique la formule base 100 du cours (Slide 17) :

$$\text{Index}_{\text{régional}, t} = 100 \times \exp\!\left(\log P_t^{\text{net}} - \log P_{t_0}^{\text{net}}\right)$$

Cela donne une vision synthétique du marché normand, indépendante de la composition des biens vendus (**qualité constante**, Slide 13).

#### Visualisation produite

Deux panneaux côte à côte :

- **Gauche** : courbes superposées de $\text{Index}_{\text{régional},t}$ pour les Maisons et les Appartements — permet de lire directement quel marché a le plus progressé depuis $t_0$.
- **Droite** : écart (spread) Maisons − Appartements, avec remplissage coloré indiquant quel type de bien surperforme à chaque période.

| Sortie | Description |
|---|---|
| `evolution_regionale_maison_vs_appart.png` | Courbes comparatives + spread |

#### Tableau récapitulatif régional

Pour chaque type de bien, le bloc affiche en console :

| Indicateur | Formule |
|---|---|
| Index final | $\text{Index}_{\text{régional}, t_{\max}}$ |
| Variation totale | $(\text{Index}_{t_{\max}} / \text{Index}_{t_{\min}} - 1) \times 100$ |
| CAGR annualisé | $(\text{Index}_{t_{\max}} / 100)^{1/n} - 1$ |

## ⚠️ Note sur les données insuffisantes dans la carte des appartements

Sur la carte choroplèthe des **appartements**, un nombre important de communes apparaît en **gris clair** (mention *"Données insuffisantes"*). Ce résultat est **normal, attendu, et cohérent avec la méthodologie du cours**.

### Pourquoi ce phénomène ?

**La Normandie est une région à dominante rurale.** Les appartements se concentrent dans quelques pôles urbains — Rouen, Caen, Le Havre, Cherbourg, Évreux — tandis que la grande majorité des ~1 500 communes normandes sont de petits villages où il ne se vend quasiment aucun appartement sur la période 2019–2024.

La procédure de **complétion des périodes manquantes** (Slide 16) comble les *trous temporels* dans une série déjà existante par interpolation linéaire. Mais elle **ne peut pas créer une série pour une commune qui n'a jamais enregistré aucune vente d'appartement** : sans aucun $\log P_i^{\text{net}}$ à agréger, aucun indice ne peut être calculé, et la commune reste grise sur la carte.

À l'inverse, pour les **maisons**, la couverture spatiale est bien plus dense car les maisons sont vendues dans presque toutes les communes, y compris les plus rurales.

### Lien avec le cours (Slides 11 et 16)

Deux mécanismes du cours expliquent directement ce phénomène :

- **Slide 11** — le seuil `MIN_OBS_COMMUNE = 30` impose qu'une commune dispose d'au moins 30 transactions pour que son effet fixe $\mu_c$ soit identifiable sans risque de sur-ajustement. Pour les appartements, seule une minorité de communes dépasse ce seuil.
- **Slide 16** — la procédure de complétion part de la grille $\{(c,t)\}$ des communes *qui ont déjà au moins une observation*. Les communes sans aucune transaction appartement ne figurent tout simplement pas dans cette grille.

Le gris sur la carte n'est donc pas un bug : c'est le **reflet honnête de la réalité du marché immobilier normand**, et une conséquence directe des choix méthodologiques du cours.

### Pistes d'amélioration (extensions possibles)

Si l'on souhaitait améliorer la couverture spatiale pour les appartements, deux options méthodologiques existent :

| Option | Avantage | Limite |
|---|---|---|
| Abaisser `MIN_OBS_COMMUNE` | Plus de communes couvertes | Risque de sur-ajustement de $\mu_c$ (Slide 11) |
| Agréger à l'échelle des **EPCI** (intercommunalités) | Effectifs suffisants partout | Perte de granularité spatiale |

Ces pistes constituent des extensions méthodologiques pertinentes à mentionner en soutenance.


---

## 🔗 Références

- **Cours** : *UE Conduite de Projet*, M1 SE – Université de Strasbourg (2025/26)
- **Données** : [DVF+ open-data](https://files.data.gouv.fr/geo-dvf/latest/csv/) — Direction Générale des Finances Publiques
- **Géométries** : [geo.api.gouv.fr](https://geo.api.gouv.fr) — API officielle des communes françaises
- **Méthodologie** : Rosen, S. (1974). *Hedonic Prices and Implicit Markets*. Journal of Political Economy.
