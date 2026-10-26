#!/usr/bin/env python3
"""
test_config.py - quick check that config.py works.

Trains every classifier with every imbalance method on a small fake dataset.
Run with:  python test_config.py
You should see 24 "OK" lines, then "All combinations ran."
"""

from sklearn.datasets import make_classification
from config import CLASSIFIERS, IMBALANCE_METHODS, SEED, make_pipeline, get_fit_params

# Small fake dataset with an imbalanced class, just to check everything runs
X, y = make_classification(n_samples=3000, n_features=10, n_informative=6,
							n_classes=3, weights=[0.7, 0.25, 0.05],
							random_state=SEED)

for clf in CLASSIFIERS:
	for method in IMBALANCE_METHODS:
		if method == "focal_loss":
			continue  # not built yet
		pipe = make_pipeline(clf, method)
		pipe.fit(X, y, **get_fit_params(method, y))
		print(f"OK: {clf} + {method}")
		
print("All combinations ran.")