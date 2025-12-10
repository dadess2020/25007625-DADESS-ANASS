<img src="Image Hatim.jpeg" style="height:464px;margin-right:432px"/>

DADESS ANASS CAC 1 
25007625




# 📊 Compte Rendu d'Analyse Exploratoire des Données (EDA)

## 🎯 Objectif
L'objectif de cette analyse exploratoire était de comprendre la structure, les distributions, et les relations entre les variables clés de la base de données des séries web (`top_rated_2000webseries.csv`) : la **note (`rating`)**, la **popularité (`popularity`)**, le **nombre de votes (`votes`)** et le **pays d'origine (`country_origin`)**.

## 1. Préparation et Nettoyage des Données

1.  **Chargement des Données :** Le fichier CSV a été chargé dans un DataFrame Pandas.
2.  **Conversion de Types :** Les colonnes `rating`, `popularity`, et `votes` ont été converties au format numérique (float) pour permettre les calculs statistiques et les visualisations.
3.  **Gestion des Valeurs Manquantes :** Les lignes avec des valeurs manquantes dans les colonnes clés utilisées pour le traçage (`rating`, `popularity`, `votes`, `country_origin`) ont été supprimées pour garantir l'intégrité des graphiques.

---

## 2. Visualisations et Observations Clés

### A. Carte de Chaleur de Corrélation

| Caractéristique | Description |
| :--- | :--- |
| **Objectif** | Mesurer et visualiser la force et la direction de la relation linéaire entre les variables numériques. |
| **Observations** | Une corrélation positive notable existe entre le **Rating** et le nombre de **Votes**. Cela indique que les séries avec une note élevée tendent également à avoir un grand nombre de votes, ce qui atteste d'un consensus de qualité auprès d'une large audience. La corrélation avec la **Popularity** est présente, mais plus faible. |
<img src="Capture d'écran 2025-12-10 155911.png" style="height:464px;margin-right:432px"/>

### B. Histogrammes de Distribution

| Caractéristique | Description |
| :--- | :--- |
| **Objectif** | Afficher la fréquence et la forme de la distribution de chaque variable numérique. |
| **Notes (`Rating`)** | La distribution est concentrée sur les valeurs élevées (principalement entre 7.0 et 8.0). Ceci est attendu car le dataset est une liste de séries "les mieux notées". |
| **Popularité et Votes** | Ces deux distributions présentent une **forte asymétrie positive (biais à droite)**. Une majorité écrasante des séries ont une popularité et un nombre de votes faibles, tandis qu'une minorité d'œuvres très célèbres se situent dans la longue queue de la distribution. |
<img src="Capture d'écran 2025-12-10 155928.png" style="height:464px;margin-right:432px"/>

### C. Diagramme de Comptage par Pays d'Origine (Top 10)

| Caractéristique | Description |
| :--- | :--- |
| **Objectif** | Compter le nombre de séries par pays d'origine pour identifier les contributeurs dominants. |
| **Résultat** | Les **États-Unis** sont le pays dominant dans ce classement des séries les mieux notées. Le **Japon** et le **Royaume-Uni** suivent, reflétant les principaux pôles de production de séries mondialement reconnues. |
<img src="Capture d'écran 2025-12-10 155938.png" style="height:464px;margin-right:432px"/>
<img src="Capture d'écran 2025-12-10 155952.png" style="height:464px;margin-right:432px"/>
<img src="Capture d'écran 2025-12-10 160014.png" style="height:464px;margin-right:432px"/>
### D. Pair Plot (Nuages de Points Croisés)

| Caractéristique | Description |
| :--- | :--- |
| **Objectif** | Visualiser simultanément les relations bivariées entre les trois variables numériques et leurs distributions. |
| **Confirmation** | Les graphiques de nuages de points confirment que les séries avec un nombre de votes très élevé tendent à maintenir un niveau de note élevé. La zone de haute popularité est clairsemée mais souvent associée à des notes solides. |

---

## 📝 Synthèse

L'analyse exploratoire confirme que la base de données est biaisée vers la qualité (notes élevées) et dominée par les productions de quelques pays, notamment les **États-Unis**. Les indicateurs de volume (`votes` et `popularity`) suivent une loi de puissance typique, où une petite fraction des séries détient la majorité de l'attention et des interactions.
