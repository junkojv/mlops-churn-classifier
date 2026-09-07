# Churn Classifier with Pipelines & MLflow

Projet complet de Machine Learning Operations (MLOps) implémentant un pipeline de classification tabulaire robuste avec `scikit-learn` (`Pipeline` + `ColumnTransformer`) et un suivi intégral des expériences avec **MLflow** (paramètres, métriques, artefacts et Model Registry).

---

## Vue d'ensemble du projet

* **Problématique Business :** Prédire le désabonnement client (*Telco Customer Churn*).
* **Dataset :** [Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (7 043 observations, 20 variables numériques et catégorielles).
* **Architecture MLOps :**
  * Prétraitement avec `ColumnTransformer` (médiane & standardisation pour les variables numériques, imputation fréquentielle & encodage One-Hot pour les variables catégorielles).
  * Optimisation d'hyperparamètres avec `GridSearchCV` et validation croisée stratifiée à 5 blocs (`StratifiedKFold`).
  * Traçabilité automatique des runs, des métriques et des artifacts avec **MLflow Tracking**.
  * Versioning et enregistrement du meilleur modèle dans le **MLflow Model Registry** (`ChurnClassifier`).
  * Automatisation complète des commandes via un **`Makefile`**.
  * Qualité du code assurée par des tests unitaires (`pytest`) et un linter (`ruff`).

---

## Structure du projet (Repo layout)

```text
mlops-project/
├── data/
│   ├── raw.csv                  # Données brutes (gitignored)
│   └── processed/
│       └── test.csv             # Jeu de test isolé pour l'évaluation (gitignored)
├── configs/
│   └── config.yaml              # Fichier de configuration (variables, hyperparamètres, CV)
├── src/
│   ├── pipeline.py              # Construction du ColumnTransformer + Modèle
│   ├── train.py                 # Entraînement, GridSearchCV et enregistrement MLflow
│   ├── evaluate.py              # Évaluation sur le jeu de test et génération des artefacts
│   ├── eda.py                   # Analyse exploratoire des données (EDA)
│   └── utils.py                 # Fonctions utilitaires (chargement YAML, nettoyage, split)
├── tests/
│   └── test_pipeline.py         # Tests unitaires pour valider le pipeline
├── artifacts/                   # Graphiques d'évaluation (matrice de confusion, ROC, PR)
├── reports/
│   └── eda/                     # Graphiques EDA générés par src/eda.py
├── Makefile                     # Automatisation (init, data, eda, train, evaluate, test, lint, ui)
├── requirements.txt             # Dépendances Python
├── pyproject.toml               # Configuration du projet et de ruff/pytest
├── .env.example                 # Variables d'environnement pour MLflow
├── .gitignore                   # Fichiers et dossiers exclus du versioning Git
└── README.md                    # Documentation complète du projet
```

---

## Installation & Démarrage rapide

Toutes les opérations sont automatisées via le `Makefile`.

### 1. Initialiser l'environnement virtuel et installer les dépendances
```bash
make init
```

### 2. Télécharger le dataset (Telco Customer Churn)
Le fichier de données n'est pas versionné (gitignored). Téléchargez-le automatiquement avec :
```bash
make data
```
Cela télécharge `data/raw.csv` depuis le dépôt public IBM.

### 3. Exécuter les tests unitaires et le linter
```bash
make test
make lint
```

### 4. Explorer les données (EDA)
Génère des graphiques d'analyse dans `reports/eda/` (distribution du churn, corrélations, taux de churn par segment) :
```bash
make eda
```

### 5. Entraîner et optimiser le modèle
Lance la recherche par grille (`GridSearchCV`), enregistre les paramètres optimaux et publie le modèle dans le registre MLflow :
```bash
make train
```

### 6. Évaluer le modèle sur le jeu de test
Calcule les métriques finales (Accuracy, Precision, Recall, F1, ROC-AUC) et génère les graphiques d'analyse dans `artifacts/` :
```bash
make evaluate
```

---

## Résultats et Performances

Sur le jeu de test indépendant (`data/processed/test.csv`) :

| Métrique | Score |
| :--- | :--- |
| **ROC-AUC** | **0.8411** |
| **Accuracy** | **80.48 %** |
| **F1-Score** | **60.32 %** |
| **Precision** | **65.52 %** |
| **Recall** | **55.88 %** |

### Artefacts générés dans MLflow :
* **`confusion_matrix.png`** : Matrice de confusion évaluant les vrais/faux positifs et négatifs.
* **`roc_curve.png`** : Courbe ROC illustrant le compromis sensibilité / spécificité.
* **`pr_curve.png`** : Courbe Précision-Rappel adaptée aux classes déséquilibrées.
* **`predictions_sample.csv`** : Échantillon des prédictions pour l'analyse d'erreurs.

---

## Visualiser dans l'interface MLflow UI

Pour explorer les runs, comparer les métriques et inspecter les artefacts dans votre navigateur :

```bash
make ui
```

Puis ouvrez votre navigateur à l'adresse : **`http://localhost:5000`** (ou `http://127.0.0.1:5000`).

Dans l'interface :
* Cliquez sur l'expérience **`churn-exp`** pour voir l'historique des runs d'entraînement et d'évaluation.
* Cliquez sur l'onglet **Models** pour visualiser le modèle enregistré **`ChurnClassifier`**.
