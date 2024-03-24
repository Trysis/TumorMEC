import os

import src.utils.constantes as cst
import src.processing.dataloader as load
import src.models.models as models
import src.models.scorer as scorer
import src.utils.auxiliary as auxiliary

# TODO : Save Model

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
N_ITER = 2  # RandomSearch settings sampling number
SCORING = {
    "accuracy": scorer.accuracy_score(to_scorer=True),
    "balanced_accuracy": scorer.balanced_accuracy_score(to_scorer=True),
    "precision": scorer.precision_score(to_scorer=True),
    "recall": scorer.recall_score(to_scorer=True),
    "auc": scorer.roc_auc_score(to_scorer=True),
    "mcc": scorer.matthews_corrcoef(to_scorer=True),
    "f1": scorer.f1_score(to_scorer=True),
}

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

filename = loader.filename_from_mask()
rootname, ext = os.path.splitext(filename)

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
        search = models.random_forest_search(
            x=x_train, y=y_train.ravel(), groups=groups_train,
            n_split=CV, stratify=True, seed=SEED, verbose=1,
            scoring=SCORING, n_iter=N_ITER, refit="f1", n_jobs=None,
            class_weight="balanced", random_state=SEED,
        )

        # Saving directory
        loader_name = f"{rootname}_{target_column}"
        loader_dir = auxiliary.create_dir(os.path.join(OUTPUT_DIR, loader_dir), add_suffix=False)
        print(search.best_estimator_)
        print(f"{search.best_score_ = }")
        print(f"{search.best_index_ = }")
        print(f"{search.best_params_ = }")
        search.cv_results_
        exit()


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

