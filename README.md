# Churn Classifier with Pipelines & MLflow

Projet complet de Machine Learning Operations (MLOps) implémentant un pipeline de classification tabulaire robuste avec `scikit-learn` (`Pipeline` + `ColumnTransformer`), un suivi intégral des expériences avec **MLflow** (paramètres, métriques, artefacts et Model Registry) et un microservice de prédiction en temps réel avec **FastAPI** et **Docker**.

---

## Vue d'ensemble du projet

* **Problématique Business :** Prédire le désabonnement client (*Telco Customer Churn*).
* **Dataset :** [Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (7 043 observations, 20 variables numériques et catégorielles).
* **Architecture MLOps :**
  * Prétraitement avec `ColumnTransformer` (médiane & standardisation pour les variables numériques, imputation fréquentielle & encodage One-Hot pour les variables catégorielles).
  * Optimisation d'hyperparamètres avec `GridSearchCV` et validation croisée stratifiée à 5 blocs (`StratifiedKFold`).
  * Traçabilité automatique des runs, des métriques et des artifacts avec **MLflow Tracking**.
  * Versioning et enregistrement du meilleur modèle dans le **MLflow Model Registry** (`ChurnClassifier`).
  * Microservice d'inférence en temps réel avec **FastAPI** et conteneurisation **Docker**.
  * Automatisation complète des commandes via un **`Makefile`**.
  * Qualité du code assurée par des tests unitaires (`pytest`) et un linter (`ruff`).

---

## Structure du projet (Repo layout)

```text
mlops-project/
├── data/
│   ├── .gitkeep                 # Maintient la structure du dossier dans Git
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
├── service/
│   └── app.py                   # Microservice API REST FastAPI (inférence en temps réel)
├── tests/
│   ├── test_pipeline.py         # Tests unitaires pour valider le pipeline ML
│   └── test_service.py          # Tests unitaires pour l'API FastAPI
├── artifacts/                   # Graphiques d'évaluation (matrice de confusion, ROC, PR)
├── reports/
│   └── eda/                     # Graphiques EDA générés par src/eda.py
├── Dockerfile                   # Image Docker de production pour l'API
├── Makefile                     # Automatisation (init, data, eda, train, evaluate, test, lint, ui, serve)
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

### 7. Démarrer le microservice d'inférence (FastAPI)
Lance l'API REST en local sur le port 8000 :
```bash
make serve
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

---

## 🚀 Microservice API d'inférence (FastAPI)

Le projet intègre un microservice REST de production (`service/app.py`) basé sur **FastAPI** permettant de servir des prédictions en temps réel à partir du modèle enregistré dans le **MLflow Model Registry** (`ChurnClassifier`).

### 1. Démarrer le serveur API
```bash
make serve
```
Le serveur démarre sur **`http://localhost:8000`**.

### 2. Documentation interactive (Swagger UI)
Accédez à l'interface Swagger générée automatiquement pour tester interactivement l'API :
👉 **`http://localhost:8000/docs`**

### 3. Vérifier l'état de santé du service (`GET /health`)
```bash
curl http://localhost:8000/health
```
*Réponse :*
```json
{"status": "ok", "service": "ChurnClassifier"}
```

### 4. Faire une prédiction de Churn en temps réel (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "tenure": 12,
       "MonthlyCharges": 70.35,
       "TotalCharges": 840.0,
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "No",
       "Dependents": "No",
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "Fiber optic",
       "OnlineSecurity": "No",
       "OnlineBackup": "No",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "No",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check"
     }'
```
*Réponse type :*
```json
{
  "churn_prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.6421
}
```

---

## 🐳 Déploiement avec Docker

Le projet est entièrement conteneurisé pour servir l'API en production :

```bash
# 1. Construire l'image Docker
docker build -t churn-classifier-api .

# 2. Lancer le conteneur
docker run -d -p 8000:8000 churn-classifier-api
```

L'API et la documentation Swagger sont alors directement opérationnelles sur **`http://localhost:8000/docs`**.
