# GRAND GUIDE ANATOMIE D’UN PROJET DATA SCIENCE – PRÉDICTION DES PRIX AAVE (CRYPTO)

---

## 1. Contexte général du projet

Ce projet s’inscrit dans le cadre de l’analyse quantitative des marchés financiers numériques, plus précisément celui des **cryptomonnaies**, qui constituent aujourd’hui un champ d’étude majeur en Data Science. Contrairement aux marchés financiers traditionnels, le marché crypto se caractérise par une **volatilité extrême**, une activité continue 24h/24 et une forte dépendance aux comportements spéculatifs.

L’actif étudié est **AAVE**, un token de la finance décentralisée (DeFi), utilisé à la fois comme moyen d’échange et comme instrument de gouvernance. L’objectif du projet est de démontrer comment un **pipeline complet de Machine Learning** peut être mis en œuvre pour exploiter des données historiques et produire une prédiction du **prix de clôture journalier**.

Au-delà du simple calcul de prédictions, ce travail vise à expliquer chaque choix méthodologique (nettoyage, features, modèle) afin de fournir une compréhension claire et reproductible du processus Data Science.

---

## 2. Présentation du dataset (source Kaggle)

Le dataset provient de Kaggle : *Top 50 Cryptocurrency Dataset*. Il regroupe les données historiques quotidiennes des principales cryptomonnaies. Dans ce projet, seul l’actif **AAVE-USD** est extrait et analysé.

### Structure des données

| Variable | Description |
|--------|------------|
| Date | Date de cotation |
| Open | Prix d’ouverture (USD) |
| High | Prix le plus élevé de la journée |
| Low | Prix le plus bas de la journée |
| Close | Prix de clôture |
| Volume | Volume échangé |

**Période couverte** : 02/10/2020 – 05/01/2026  
**Nombre d’observations** : 1923 lignes  
**Type de données** : Série temporelle financière

---

## 3. Problématique métier

Dans un contexte d’investissement crypto, la capacité à anticiper le prix de clôture permet :
- une meilleure prise de décision d’achat/vente,
- une réduction du risque lié à la volatilité,
- une automatisation des stratégies de trading.

La variable cible est **Close**, tandis que les variables explicatives sont les prix **Open, High, Low** et le **Volume**.

---

## 4. Chargement et nettoyage des données

La phase de chargement et de nettoyage constitue une étape critique, car la qualité des résultats dépend directement de la qualité des données utilisées. Le fichier CSV issu de Kaggle présente une anomalie au niveau de l’en-tête, ce qui nécessite une manipulation spécifique lors de l’importation.

### Code Python – Chargement et préparation

```python
import pandas as pd

# Chargement du fichier CSV
# skiprows=1 permet d’ignorer la première ligne corrompue
df = pd.read_csv('aave.csv', skiprows=1)

# Renommage explicite des colonnes
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Conversion de la colonne Date en format datetime
df['Date'] = pd.to_datetime(df['Date'])

# Tri chronologique indispensable pour les séries temporelles
df = df.sort_values('Date').reset_index(drop=True)

# Conversion des variables numériques
df[['Close', 'High', 'Low', 'Open', 'Volume']] = df[['Close', 'High', 'Low', 'Open', 'Volume']].apply(pd.to_numeric, errors='coerce')

# Suppression des valeurs manquantes
df.dropna(inplace=True)
```

Ce nettoyage garantit l’absence d’erreurs de type et assure la cohérence temporelle avant toute analyse statistique ou modélisation.

---

## 5. Feature Engineering

Le Feature Engineering consiste à transformer les données brutes en variables plus informatives pour le modèle. Dans le cadre des marchés financiers, il est essentiel de capter non seulement le niveau des prix, mais également leur **variation** et leur **intensité d’échange**.

### Code Python – Création des variables dérivées

```python
# Rendement journalier du prix de clôture
df['Return'] = df['Close'].pct_change()

# Amplitude relative journalière (mesure de volatilité)
df['RangePct'] = (df['High'] - df['Low']) / df['Close']

# Volume échangé exprimé en dollars
df['VolumeUSD'] = df['Close'] * df['Volume']

# Suppression des premières lignes avec NaN générées par pct_change
df.dropna(inplace=True)
```

Ces nouvelles variables permettent de mieux représenter la dynamique du marché, en intégrant à la fois la variation des prix et la pression des volumes.

-------|--------|---------------|
| Return | Close.pct_change() | Rendement journalier |
| RangePct | (High − Low) / Close | Volatilité journalière |
| VolumeUSD | Close × Volume | Volume échangé en USD |

Ces variables permettent de mieux capturer la dynamique du marché.

---

## 6. Analyse exploratoire des données (EDA)

### Statistiques descriptives

| Variable | Moyenne | Écart-type | Min | Max |
|--------|--------|-----------|-----|-----|
| Close | ≈ 111 | ≈ 111 | 0.52 | 632.27 |
| Volume | ≈ 2.85e8 | élevé | faible | très élevé |

Les résultats montrent une **volatilité extrême**, typique des cryptomonnaies, avec des variations de prix très importantes.

### 6.1 Évolution temporelle du prix de clôture

Le premier graphique représente l’évolution du prix de clôture (**Close**) d’AAVE au cours du temps.

**Interprétation** :
- Forte tendance haussière entre 2020 et 2021 (bull run crypto).
- Phase de correction marquée en 2022.
- Période de consolidation et de volatilité modérée entre 2023 et 2026.

Ce graphique permet d’identifier clairement les cycles du marché.

### Code Python – Prix de clôture dans le temps

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Close'])
plt.title('Évolution du prix de clôture AAVE')
plt.xlabel('Date')
plt.ylabel('Prix (USD)')
plt.show()
```

---


---
### 6.2 Matrice de corrélation

La matrice de corrélation permet de mesurer les relations linéaires entre les variables numériques.

**Interprétation** :
- Corrélation très forte entre Open, High, Low et Close (> 0.99).
- VolumeUSD moins corrélé au prix, mais utile comme variable contextuelle.

### Code Python – Heatmap de corrélation

```python
import seaborn as sns
import numpy as np

plt.figure(figsize=(8,6))
corr = df[['Open', 'High', 'Low', 'Close', 'VolumeUSD']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de corrélation')
plt.show()
```

--------|--------|-----------|-----|-----|
| Close | ≈ 111 | ≈ 111 | 0.52 | 632.27 |
| Volume | ≈ 2.85e8 | élevé | faible | très élevé |

Les résultats montrent une **volatilité extrême**, typique des cryptomonnaies, avec des variations de prix très importantes.

### Corrélations

Les variables **Open, High, Low et Close** présentent des corrélations supérieures à **0.99**, ce qui justifie l’utilisation d’un modèle linéaire comme baseline.

---

## 7. Méthodologie de découpage des données

Étant donné la nature temporelle des données, un **split chronologique** est appliqué :
- 80 % pour l’entraînement
- 20 % pour le test

Cette approche évite le **data leakage**, erreur fréquente dans les projets de séries temporelles.

---

## 8. Modélisation – Régression Linéaire Multiple

La régression linéaire multiple est utilisée comme modèle de référence (baseline). Elle cherche à expliquer la variable cible **Close** comme une combinaison linéaire des variables explicatives.

### Code Python – Entraînement du modèle

```python
from sklearn.linear_model import LinearRegression

# Séparation des variables explicatives et de la cible
X = df[['Open', 'High', 'Low', 'VolumeUSD']]
y = df['Close']

# Découpage chronologique (80% train, 20% test)
split_idx = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Initialisation et entraînement du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur l’échantillon de test
y_pred = model.predict(X_test)
```

Ce modèle sert de point de comparaison pour des approches plus avancées et permet une interprétation directe des relations entre variables.

----|----------|
| X | Open, High, Low, VolumeUSD |
| y | Close |

---

## 9. Évaluation du modèle

L’évaluation du modèle permet de mesurer sa capacité à expliquer les variations du prix de clôture.

### Code Python – Métriques de performance

```python
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Calcul des métriques
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² : {r2:.4f}")
print(f"RMSE : {rmse:.2f}")
```

Un **R² proche de 1** indique que le modèle explique presque entièrement la variance du prix de clôture, ce qui est cohérent avec la forte corrélation entre les variables OHLC.

### Limites

Ces performances élevées doivent être interprétées avec prudence : le modèle repose sur des informations très proches de la variable cible et ne prend pas en compte les facteurs exogènes (actualités, sentiment de marché, corrélation inter‑crypto).

---

## Conclusion générale

Ce projet démontre de manière détaillée comment appliquer les méthodes de Data Science à un problème réel de prédiction financière. Il met en évidence l’importance du nettoyage des données, du Feature Engineering et du choix méthodologique du modèle.

Malgré ses limites, cette approche constitue une base solide pour des extensions futures intégrant des modèles de séries temporelles avancés et des indicateurs techniques plus sophistiqués.
