#!/usr/bin/env python3
"""
config.py - shared settings for the CIS 4930 intrusion detection project.

Everyone imports from this file so we all use the same seeds, models and
imbalance methods. If you need to change a setting, change it HERE and tell
the group, instead of hard-coding your own value in your experiment file.

Usage:
	from config import SEED, RARE_FAMILIES, make_pipeline, get_fit_params
"""

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


SEED = 42                      # use this everywhere a random_state is needed
N_FOLDS = 5                    # stratified cross-validation folds
SUBSAMPLE_SEEDS = [0, 1, 2]    # E2 repeats each reduction with these seeds

# Put the two official CSVs in a folder called data/
TRAIN_PATH = "data/UNSW_NB15_training-set.csv"
TEST_PATH = "data/UNSW_NB15_testing-set.csv"
RESULTS_PATH = "results/results.csv"   # everyone appends rows here

ID_COLUMN = "id"
BINARY_LABEL = "label"         # 0 = normal, 1 = attack
FAMILY_LABEL = "attack_cat"    # "Normal" or one of the nine attack families
CATEGORICAL_COLUMNS = ["proto", "service", "state"]

# The official CSVs spell it "Backdoor" (no s).
ATTACK_FAMILIES = ["Fuzzers", "Analysis", "Backdoor", "DoS", "Exploits",
					"Generic", "Reconnaissance", "Shellcode", "Worms"]
RARE_FAMILIES = ["Worms", "Shellcode", "Backdoor", "Analysis"]

# Heavily skewed count/size columns that get a log(1 + x) transform
LOG_COLUMNS = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
				"sload", "dload", "sinpkt", "dinpkt", "sjit", "djit",
				"smean", "dmean", "response_body_len"]

# Protocol values seen fewer times than this get grouped into "other"
PROTO_MIN_COUNT = 1000

CLASSIFIERS = ["decision_tree", "random_forest", "xgboost", "mlp"]
IMBALANCE_METHODS = ["none", "class_weights", "undersample",
					"smote", "adasyn", "smote_enn", "focal_loss"]

SMOTE_K = 5   # neighbors for SMOTE/ADASYN

# E2: keep this fraction of each rare family's original training records.
# Worms only has ~130 training records, so the smallest levels round to
# almost nothing. E2 keeps at least MIN_RARE_RECORDS so SMOTE can still run
# (SMOTE needs at least SMOTE_K + 1 records of a class).
SUBSAMPLE_LEVELS = [0.05, 0.01, 0.005, 0.001]
MIN_RARE_RECORDS = SMOTE_K + 1

# False alarms are reported per this many normal connections
FALSE_ALARM_SCALE = 100_000


def get_classifier(name):
	"""Return a fresh, untrained classifier with our shared settings."""
	if name == "decision_tree":
		return DecisionTreeClassifier(random_state=SEED)
	if name == "random_forest":
		return RandomForestClassifier(n_estimators=200, n_jobs=-1,
										random_state=SEED)
	if name == "xgboost":
		return XGBClassifier(n_estimators=300, tree_method="hist",
							n_jobs=-1, random_state=SEED)
	if name == "mlp":
		return MLPClassifier(hidden_layer_sizes=(64, 32),
							early_stopping=True, random_state=SEED)
	raise ValueError(f"Unknown classifier: {name}")
	
	
def get_resampler(method):
	"""Return the resampling step for a method, or "passthrough" if the
	method doesn't change the data (none and class_weights)."""
	if method in ("none", "class_weights"):
		return "passthrough"
	if method == "undersample":
		return RandomUnderSampler(random_state=SEED)
	if method == "smote":
		return SMOTE(k_neighbors=SMOTE_K, random_state=SEED)
	if method == "adasyn":
		return ADASYN(n_neighbors=SMOTE_K, random_state=SEED)
	if method == "smote_enn":
		return SMOTEENN(smote=SMOTE(k_neighbors=SMOTE_K, random_state=SEED))
	if method == "focal_loss":
		raise NotImplementedError(
			"Focal loss needs a custom XGBoost objective that isn't built yet.")
	raise ValueError(f"Unknown imbalance method: {method}")
	
	
def make_pipeline(classifier, method):
	"""Build scaler -> resampler -> classifier.

	Because this is an imblearn Pipeline, resampling only happens on the
	data passed to .fit() (the training fold), never on validation or test
	data. The scaler comes first even for tree models because SMOTE, ADASYN
	and ENN find neighbors by distance, so unscaled features would distort
	which records count as neighbors.
	"""
	if method == "focal_loss" and classifier != "xgboost":
		raise ValueError("Focal loss is only used with XGBoost.")
	return Pipeline([
		("scale", StandardScaler()),
		("resample", get_resampler(method)),
		("clf", get_classifier(classifier)),
	])
	

def get_fit_params(method, y_train):
	"""Extra arguments for pipeline.fit().

	For class_weights we pass balanced sample weights, which works the same
	way for all four classifiers (MLP needs scikit-learn 1.7 or newer).

	Example:
		pipe = make_pipeline("random_forest", "class_weights")
		pipe.fit(X_train, y_train, **get_fit_params("class_weights", y_train))
	"""
	if method == "class_weights":
		return {"clf__sample_weight": compute_sample_weight("balanced", y_train)}
	return {}
