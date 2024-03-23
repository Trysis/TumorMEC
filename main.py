import os

import src.utils.constantes as cst
import src.processing.dataloader as load
import src.models.models as models


# Generic attributes
SEED = 42
TEST_SIZE = 0.2

MAIN_DIR = "./"
OUTPUT_DIR = os.path.join(MAIN_DIR, cst.OUTPUT_DIRNAME)
DATA_DIR = os.path.join(MAIN_DIR, cst.DATA_DIRNAME)

# Attributes for DataLoader
MASK_CONDITION = [cst.WT]
MASK_TYPE = [cst.CD3]
MASK_TUMOR = [cst.IN_TUMOR]  # Only on or outside tumor
MASK_DENSITY = [cst.IN_FIBER]  # Only in or outside fiber

REMOVE_NONE = True  # True or False
REPLACE_ABERRANT = -3  # Set to None or actual value

# Attributes specifying features, target and sample
FEATURES = [cst.x_fiber_columns,]
TARGETS = [load.plus_cmask,]
TARGETS_COLNAMES = [target_col(return_key=True) for target_col in TARGETS]
SAMPLE_GROUP = ["FileName",]

# Training regimen
CV = 2

# Load data
loader = load.DataLoader(
    data_dir=DATA_DIR,
    mask_condition=MASK_CONDITION,
    mask_type=MASK_TYPE,
    mask_tumor=MASK_TUMOR,
    mask_fiber=MASK_DENSITY,
    replace_aberrant=REPLACE_ABERRANT,
    aberrant_columns=cst.aberrant_columns,
    remove_none=REMOVE_NONE
)

dataframe = loader.load_data(
    targets=TARGETS,
    type=cst.data_type,
    save=False,
    force_default=False,
)

# Define X and Y
for target_column in TARGETS_COLNAMES:
    for features_column in FEATURES:
        # Features and Target(s)
        x, y, groups = models.split_xy(
            df=dataframe, x_columns=features_column, y_columns=target_column, groups=SAMPLE_GROUP
        )
        # Train and Test
        x_train, x_test, y_train, y_test, groups_train, groups_test = models.split_data(
            x, y, groups=groups, n_splits=1, test_size=TEST_SIZE, stratify=True, seed=SEED
        )
        # Hyperparameters search
        models.random_forest_search(
            x=x_train, y=y_train, groups=groups_train,
            n_split=5, stratify=True, seed=SEED, verbose=0,
            scoring={
                    "accuracy": metrics.accuracy_score,
                    "balanced_accuracy": metrics.balanced_accuracy_score,
                    "precision": metrics.precision_score,
                    "recall": metrics.recall_score,
                    "auc": metrics.roc_auc_score,
                    "mcc": metrics.matthews_corrcoef,
                    "f1": metrics.f1_score,
            },
            n_iter=5, refit=True, n_jobs=None,
            class_weight="balanced", random_state=SEED,
            param_criterion=("entropy",),
            param_n_estimators=(20, 30, 40, 60),
            param_max_features=("sqrt,"),
            param_max_depths=(2, 4, 8, 10, 15, 20),
            param_min_s_split=(2, 4, 16, 32),
            param_min_s_leaf=(1, 5),
            param_bootstrap=(False, True),
        )
        print(f"{x_train = }\n{x_test = }")
        print(f"{y_train = }\n{y_test = }")
        print(f"{groups_train = }\n{groups_test = }")
        exit()

CROSS_VAL = 10
PREDICT_CLASS = ""

if __name__ == "__main__":
    """
    train_val (colname : train/val)
    cross_val
        matrice de confusion somme

    random_forest

    importance (permu, autre)

    explain #shap 
    https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Iris%20classification%20with%20scikit-learn.html
    https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Diabetes%20regression.html#Random-forest

    save specificite, sensibilite
    save matrice de confusion

    save auc, acc, mcc, f1
    save auc, acc, mcc, f1 plot

    save_importance
    save_importance_plot

    save_explained
    save_explained_plot
    """

