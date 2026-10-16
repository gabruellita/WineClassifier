# 🍷 Wine Quality Classifier

Proiect de **Machine Learning** care clasifică vinurile după calitate, pe baza caracteristicilor lor fizico-chimice. Scorul de calitate original (3–9) e grupat în trei clase:

| Clasă    | Scor quality |
|----------|--------------|
| `low`    | 3 – 4        |
| `medium` | 5 – 6        |
| `high`   | 7 – 9        |

Proiectul a fost realizat în **două părți**: o primă etapă de explorare a datelor și un model de bază, apoi o a doua etapă cu modele mai complexe, optimizare și deploy.

---

## 📊 Setul de date

[Wine Quality Dataset – UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality), încărcat direct din URL în notebook-uri:

- **Vin alb**: 4.898 de probe (3.961 după eliminarea duplicatelor)
- **Vin roșu**: 1.599 de probe (1.359 după eliminarea duplicatelor)

**Caracteristici (11):** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.

Principala dificultate a problemei este **dezechilibrul claselor**: clasa `medium` domină (~75% din date), în timp ce `low` are sub 5% din exemple.

---

## 📁 Structura proiectului

```
├── WineClassifier.ipynb                                   # Partea 1 – notebook
├── WineClassifier.pdf                                     # Partea 1 – raport exportat în PDF
├── ML_Models_Training,_Optimization_and_Deployment.ipynb  # Partea 2 – notebook
└── README.md
```

---

## Partea 1 – EDA și RidgeClassifier

📄 [`WineClassifier.pdf`](WineClassifier.pdf) · [`WineClassifier.ipynb`](WineClassifier.ipynb)

### Ce s-a făcut
- **Analiza exploratorie (EDA)** pentru vinul alb și pentru vinul roșu: valori lipsă, duplicate, statistici descriptive, histograme, boxplot-uri pentru outlieri, heatmap de corelații Pearson.
- **Preprocesare**: eliminarea duplicatelor, split stratificat 80/20 (train/test), standardizare cu `StandardScaler`.
- **Model de bază: `RidgeClassifier`**, ales pentru regularizarea L2, care penalizează coeficienții mari, reduce varianța și gestionează corelațiile dintre variabile.
- **Importanța caracteristicilor**, calculată din coeficienții modelului.
- **Alte încercări**: tuning pentru `alpha`, `class_weight='balanced'` și oversampling cu **ADASYN** pentru clasa rară `low`.
- **Vin roșu**: PCA, `SelectKBest` și comparație prin `GridSearchCV` (CV pe 5 fold-uri) între `RidgeClassifier`, `RandomForestClassifier` și `GradientBoostingClassifier`.

### Observații cheie din corelații
- **Alcoolul** are cea mai puternică corelație pozitivă cu calitatea (~+0.47 la ambele tipuri de vin).
- **Densitatea** (vin alb, −0.35) și **aciditatea volatilă** (vin roșu, −0.40) au cele mai puternice corelații negative.

### Rezultate
| Dataset   | Model                          | Acuratețe |
|-----------|--------------------------------|-----------|
| Vin alb   | RidgeClassifier (`alpha=1.0`)  | 77.43%    |
| Vin roșu  | RidgeClassifier (GridSearch)   | 83.13%    |
| Vin roșu  | RandomForest (GridSearch)      | 86.88%    |
| Vin roșu  | GradientBoosting (GridSearch)  | 86.88%    |

Acuratețea ridicată ascunde o problemă: modelele liniare **nu prezic deloc clasa `low`** (recall 0) din cauza dezechilibrului. De aici a pornit Partea 2.

---

## Partea 2 – Antrenare, optimizare și deploy

📓 [`ML_Models_Training,_Optimization_and_Deployment.ipynb`](ML_Models_Training,_Optimization_and_Deployment.ipynb)

Partea 2 se concentrează pe **vinul alb**, cu un split **70% train / 10% validare / 20% test** și metrici potrivite pentru clase dezechilibrate: **F1-macro, MCC (Matthews Correlation Coefficient) și ROC AUC (One-vs-Rest)**, pe lângă acuratețe.

### Modele antrenate
1. **RidgeClassifier**: baseline, apoi tuning (`alpha`, `solver`, `class_weight`) și ADASYN (`minority` / `all`).
2. **XGBoost**: boosting cu early stopping pe setul de validare, variații de `max_depth`, `subsample`, `min_child_weight`, obiectiv (`softprob` / `softmax`), ADASYN și `GridSearchCV`.
3. **MLP (Multilayer Perceptron)** în TensorFlow/Keras: rețea complet conectată (128 → 64 → 3, ReLU + Softmax, Dropout), potrivită pentru date tabulare numerice.

### Optimizarea rețelei neuronale
Optimizare manuală/empirică a hiperparametrilor:
- **Optimizator**: `Adam` vs. `SGD` cu momentum 0.9 și Nesterov
- **Dropout**: 0.3 vs. 0.5
- **Loss**: `categorical_crossentropy`
- **ADASYN** pentru echilibrarea setului de antrenare

### Tabel comparativ (setul de test)

| Model                              | Accuracy | ROC AUC    | MCC        | F1-macro |
|------------------------------------|----------|------------|------------|----------|
| RidgeClassifier (`alpha=1.0`)      | 77.64%   | 0.5582     | 0.2805     | 0.40     |
| RidgeClassifier + ADASYN           | 49.43%   | 0.6035     | —          | 0.43     |
| XGBoost                            | 77.18%   | 0.3289     | 0.3246     | 0.47     |
| XGBoost + ADASYN                   | 75.91%   | 0.8040     | 0.3246     | 0.54     |
| MLP (baseline)                     | 76.80%   | 0.8250     | 0.3214     | 0.49     |
| **MLP + Adam + Dropout 0.3 + ADASYN** | 66.20% | 0.7960     | 0.3477     | **0.56** |
| MLP + SGD Nesterov + Dropout 0.5 + ADASYN | 64.65% | **0.8113** | **0.3483** | 0.49 |

### Modelul ales pentru deploy
**MLP + Adam + Dropout 0.3 + ADASYN**: are cel mai bun **F1-macro**, deci cel mai bun echilibru între cele trei clase, un MCC ridicat și rezultate stabile între rulări. Acuratețea e mai mică decât la modelele care prezic aproape doar `medium`, dar modelul recunoaște mult mai bine clasele minoritare `low` și `high`.

### Deploy
- **Export** al modelului în format `SavedModel` (`model.export("my_model/1")`).
- **Local**: servire prin **TensorFlow Serving** (REST API pe portul 8501) și request-uri JSON de predicție.
- **Cloud**: upload în **Google Cloud Storage** și deploy pe un endpoint **Google Cloud Vertex AI**, cu predicții trimise prin SDK-ul `google-cloud-aiplatform`.

---

## 🛠️ Tehnologii

- **Python 3.12**, Jupyter Notebook / Google Colab
- **pandas**, **NumPy**: procesarea datelor
- **matplotlib**, **seaborn**: vizualizări
- **scikit-learn**: RidgeClassifier, RandomForest, GradientBoosting, GridSearchCV, metrici
- **imbalanced-learn**: ADASYN
- **XGBoost**
- **TensorFlow / Keras**: rețeaua neuronală MLP
- **TensorFlow Serving**, **Google Cloud Vertex AI**: deploy

## ▶️ Rulare

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow
jupyter notebook
```

Datele se descarcă automat de la UCI la rulare. Secțiunea de deploy din Partea 2 necesită `tensorflow-model-server` și un proiect Google Cloud propriu (`project_id`, bucket, credențiale).

## 📚 Bibliografie

- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)
- [imbalanced-learn – ADASYN](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.ADASYN.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [TensorFlow & Keras documentation](https://www.tensorflow.org)
- *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, Aurélien Géron
