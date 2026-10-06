"""Glass type classification from chemical composition (UCI Glass Identification).

Six classes: 1 building_windows_float, 2 building_windows_non_float,
3 vehicle_windows_float, 5 containers, 6 tableware, 7 headlamps.
(Class 4 does not exist in the dataset.)

Usage: python glass_classifier.py [path/to/glass.csv]
The CSV has no header: RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, Type
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42
COLS = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
NAMES = {1: "building_float", 2: "building_non_float", 3: "vehicle_float",
         5: "containers", 6: "tableware", 7: "headlamps"}

path = sys.argv[1] if len(sys.argv) > 1 else "glass.csv"
df = pd.read_csv(path, header=None, names=COLS)
X, y = df.drop(columns="Type"), df["Type"]

print(f"Rows: {len(df)}, missing values: {int(df.isna().sum().sum())}")
print("Class counts:\n", y.value_counts().sort_index().to_string(), "\n")

# Hold out a stratified test set that is never used for model selection.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)

candidates = {
    "kNN": Pipeline([("s", StandardScaler()),
                     ("m", KNeighborsClassifier(n_neighbors=3, weights="distance"))]),
    "SVM (RBF)": Pipeline([("s", StandardScaler()),
                           ("m", SVC(C=10, gamma="scale", class_weight="balanced"))]),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=SEED),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", random_state=SEED),
}

# Compare on the training split only, with repeated stratified CV.
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
print("Cross-validated accuracy on training split (5x5 CV):")
scores = {}
for name, model in candidates.items():
    s = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="accuracy")
    scores[name] = s.mean()
    print(f"  {name:15s} {s.mean():.3f} +/- {s.std():.3f}")

# Tune the two strongest families lightly.
grids = {
    "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=SEED),
                      {"n_estimators": [300, 600], "max_depth": [None, 8, 12],
                       "min_samples_leaf": [1, 2], "max_features": ["sqrt", 0.5]}),
    "Extra Trees": (ExtraTreesClassifier(class_weight="balanced", random_state=SEED),
                    {"n_estimators": [300, 600], "max_depth": [None, 8, 12],
                     "min_samples_leaf": [1, 2], "max_features": ["sqrt", 0.5]}),
}
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
tuned = {}
print("\nTuning:")
for name, (est, grid) in grids.items():
    gs = GridSearchCV(est, grid, cv=inner, scoring="accuracy", n_jobs=-1).fit(X_tr, y_tr)
    tuned[name] = gs
    print(f"  {name:15s} best CV acc {gs.best_score_:.3f}  params {gs.best_params_}")

best_name = max(tuned, key=lambda k: tuned[k].best_score_)
best = tuned[best_name].best_estimator_
print(f"\nSelected model: {best_name}")

# Final, single evaluation on the untouched test set.
pred = best.predict(X_te)
acc = (pred == y_te).mean()
print(f"Test accuracy: {acc:.3f} on {len(y_te)} samples\n")
labels = sorted(NAMES)
print(classification_report(y_te, pred, labels=labels,
                            target_names=[NAMES[l] for l in labels], zero_division=0))
cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels),
                  index=[f"true_{l}" for l in labels],
                  columns=[f"pred_{l}" for l in labels])
print("Confusion matrix:\n", cm.to_string(), "\n")

imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importances:\n", imp.round(3).to_string())
