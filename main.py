import os

import src.utils.constantes as cst
import src.processing.dataloader as load
import src.models.models as models
import src.models.scorer as scorer
import src.utils.auxiliary as auxiliary
import src.utils.summary as summary

import numpy as np
import pandas as pd

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
FIT_WITH="f1"
TARGETS_WEIGHTS="balanced"
## Hyperparameters search
hsearch_criterion=["entropy",]
hsearch_n_estimators=[20, 30, 40, 60]
hsearch_max_features=["sqrt"]
hsearch_max_depths=[2, 4, 8, 10, 15, 20]
hsearch_min_s_split=[2, 4, 16, 32]
hsearch_min_s_leaf=[1, 5]
hsearch_bootstrap=[False, True]

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
        # Saving output
        loader_name = f"{rootname}_{target_column}"
        loader_dir = auxiliary.create_dir(os.path.join(OUTPUT_DIR, loader_name), add_suffix=False)
        summary_file = os.path.join(loader_dir, "summary.txt")
        hsearch_file = os.path.join(loader_dir, "search_param.csv")
        e = summary.summarize(
            summary.mapped_summary({
                "Condition": [c.name for c in MASK_CONDITION],
                "Type": [t.name for t in MASK_TYPE],
                "Tumor": [t.name for t in MASK_TUMOR],
                "Fiber": [d.name for d in MASK_DENSITY],
                "Remove none": REMOVE_NONE,
                "Replace aberrant": REPLACE_ABERRANT,
            }, map_sep=":"),
            title="Parameters",
            filepath=summary_file, mode="w"
        )
        f = summary.summarize(
            summary.mapped_summary({
                "MODEL": models.ESTIMATOR,
                "Cross-valdiation N-Folds": CV,
                "RandomSearch N-iter": N_ITER,
                "Select best model with": FIT_WITH
            }, map_sep=":"),
            summary.arg_summary("Scoring", "\n" + summary.mapped_summary(SCORING, padding_left=4), new_line=False),
            summary.arg_summary(
                "Hyperparameters search",
                "\n" +
                summary.mapped_summary({
                    "Criterion": hsearch_criterion,
                    "N-Tree": hsearch_n_estimators,
                    "N-Features": hsearch_max_features,
                    "Max depths": hsearch_max_depths,
                    "Min sample split": hsearch_min_s_split,
                    "Min sample leaf": hsearch_min_s_leaf,
                    "Bootstrap": hsearch_bootstrap
                }, map_sep="=", padding_left=4),
                new_line=False
            ),
            subtitle="Training regiment",
            filepath=summary_file
        )
        # Features and Target(s)
        x, y, groups = models.split_xy(
            df=dataframe, x_columns=features_column, y_columns=target_column, groups=SAMPLE_GROUP
        )

        label_groups = pd.DataFrame(dataframe[SAMPLE_GROUP].agg(';'.join, axis=1), columns=["label"])
        df_mapped_groups = pd.concat([label_groups, pd.DataFrame({"groups": groups})], axis=1).drop_duplicates()    
        mapped_groups = dict(zip(df_mapped_groups.label, df_mapped_groups.groups))

        a = summary.df_summary(
            x=x, y=y,
            unique_groups=np.unique(groups),
            x_columns=features_column,
            y_columns=target_column,
            groups_columns=SAMPLE_GROUP,
            mapped_groups=mapped_groups,
            new_line=True,
            filepath=summary_file
        )

        # Train and Test
        x_train, x_test, y_train, y_test, groups_train, groups_test = models.split_data(
            x, y, groups=groups, n_splits=1, test_size=TEST_SIZE, stratify=True, seed=SEED
        )
        summary.summarize(
            b := summary.xy_summary(
                x_train, y_train, unique_groups=np.unique(groups_train), title="Train",
                x_label="x_train shape", y_label="y_train shape", groups_label="groups_train",
            ),
            c := summary.xy_summary(
                x_test, y_test, unique_groups=np.unique(groups_test), title="Test",
                x_label="x_test shape", y_label="y_test shape", groups_label="groups_test",
            ),
            filepath=summary_file
        )

        # Hyperparameters search
        search = models.random_forest_search(
            x=x_train, y=y_train.ravel(), groups=groups_train,
            n_split=CV, stratify=True, seed=SEED, verbose=1,
            scoring=SCORING, n_iter=N_ITER, refit=FIT_WITH, n_jobs=None,
            class_weight=TARGETS_WEIGHTS, random_state=SEED,
            param_criterion=hsearch_criterion,
            param_n_estimators=hsearch_n_estimators,
            param_max_features=hsearch_max_features,
            param_max_depths=hsearch_max_depths,
            param_min_s_split=hsearch_min_s_split,
            param_min_s_leaf=hsearch_min_s_leaf,
            param_bootstrap=hsearch_bootstrap,
        )
        # Save tested parameters
        pd.DataFrame(search.cv_results_).to_csv(hsearch_file)
        print(search.best_estimator_)
        print(f"{search.best_score_ = }")
        print(f"{search.best_index_ = }")
        print(f"{search.best_params_ = }")
        
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

