# 🌸 Iris Flower Classification

Multi-class classification comparing **KNN**, **SVM**, and **Decision Tree** with full EDA pipeline.

## Results

| Model | CV Accuracy |
|-------|------------|
| K-Nearest Neighbours (k=5) | ~96% |
| Support Vector Machine (RBF) | ~97% |
| Decision Tree (max_depth=4) | ~95% |

## Highlights
- Full exploratory data analysis: pairplots, correlation heatmaps
- StandardScaler preprocessing + 5-fold cross-validation
- Comparison of 3 classifiers on accuracy, precision, recall, F1
- Decision boundary visualisation

## Run

```bash
pip install -r requirements.txt
python iris_classifier.py
```

## Tech Stack
`Python` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`
