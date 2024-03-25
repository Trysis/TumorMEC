import os

import src.utils.constantes as cst
import src.utils.summary as summary
import src.utils.auxiliary as auxiliary
import src.processing.dataloader as load
import src.models.scorer as scorer
import src.models.models as models
import src.models.display_results as display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO : Save Model

# Generic attributes
SEED = 42
VAL_SIZE = 0.2
TEST_SIZE = 0.1

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
FEATURES = {"loc-fract": cst.x_fiber_columns}
TARGETS = [load.plus_cmask, load.enrich_cmask, load.enrich_2_cmask]
TARGETS_COLNAMES = [target_col(return_key=True) for target_col in TARGETS]
SAMPLE_GROUP = ["FileName",]
REMOVE_SAMPLE = {"FileName": []}  # TODO

# Training regimen
CV = 2
N_ITER = 1  # RandomSearch settings sampling number
CV_TRAIN = True
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
    for key, features_column in FEATURES.items():
        # Saving output
        loader_name = f"{rootname}_{key}_{target_column}"  # MAIN DIR
        # dir
        loader_dir = auxiliary.create_dir(os.path.join(OUTPUT_DIR, loader_name), add_suffix=False)
        cv_dir = auxiliary.create_dir(os.path.join(loader_dir, "cv"), add_suffix=False)
        cv_plot_dir = auxiliary.create_dir(os.path.join(cv_dir, "plots"), add_suffix=False)
        importance_dir = auxiliary.create_dir(os.path.join(loader_dir, "importance"), add_suffix=False)
        importance_plot_dir = auxiliary.create_dir(os.path.join(importance_dir, "plots"), add_suffix=False)
        # files
        summary_file = os.path.join(loader_dir, "summary.txt")
        hsearch_file = os.path.join(loader_dir, "search_param.csv")
        hsearch_train_file = os.path.join(cv_dir, "search_param-reduced_train.csv")
        hsearch_val_file = os.path.join(cv_dir, "search_param-reduced_val.csv")
        cv_scores_train_file = os.path.join(cv_plot_dir, "cv_scores_train.png")
        cv_scores_val_file = os.path.join(cv_plot_dir, "cv_scores_val.png")
        cv_cfmatrix_train_file = os.path.join(cv_plot_dir, "cv_cfmatrix_train.png")
        cv_cfmatrix_val_file = os.path.join(cv_plot_dir, "cv_cfmatrix_val.png")
        mdi_importance_file = os.path.join(importance_dir, "mean-decrease-impurity.csv")
        permut_importance_train_file = os.path.join(importance_dir, "permutation_train.csv")
        permut_importance_test_file = os.path.join(importance_dir, "permutation_test.csv")
        boruta_importance_train_file = os.path.join(importance_dir, "boruta_train.csv")
        boruta_importance_test_file = os.path.join(importance_dir, "boruta_test.csv")
        mdi_plot_file = os.path.join(importance_plot_dir, "mean-decrease-impurity.png")
        permut_plot_train_file = os.path.join(importance_plot_dir, "permutation_train.png")
        permut_plot_test_file = os.path.join(importance_plot_dir, "permutation_test.png")
        boruta_plot_train_file = os.path.join(importance_plot_dir, "boruta_train.png")
        boruta_plot_test_file = os.path.join(importance_plot_dir, "boruta_test.png")

        summary.summarize(
            summary.mapped_summary({
                "Condition": [c.name for c in MASK_CONDITION],
                "Type": [t.name for t in MASK_TYPE],
                "Tumor": [t.name for t in MASK_TUMOR],
                "Fiber": [d.name for d in MASK_DENSITY],
                "Remove none": REMOVE_NONE,
                "Replace aberrant": REPLACE_ABERRANT,
                "SEED": SEED,
            }, map_sep=":"),
            title="Parameters",
            filepath=summary_file, mode="w"
        )
        summary.summarize(
            summary.mapped_summary({
                "MODEL": models.ESTIMATOR,
                "TEST RATIO": TEST_SIZE,
                "GROUPS": True if SAMPLE_GROUP else False,
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

        summary.df_summary(
            x=x, y=y,
            unique_groups=np.unique(groups),
            x_columns=features_column,
            y_columns=target_column,
            groups_columns=SAMPLE_GROUP,
            mapped_groups=mapped_groups,
            new_line=True,
            filepath=summary_file
        )

        # Train, Val, Test
        x_train, x_val_test, y_train, y_val_test, groups_train, groups_val_test = models.split_data(
            x, y, groups=groups, n_splits=1, test_size=VAL_SIZE+TEST_SIZE, stratify=True, seed=SEED
        )
        x_val, x_test, y_val, y_test, groups_val, groups_test = models.split_data(
            x_val_test, y_val_test, groups=groups_val_test, n_splits=1,
            test_size=TEST_SIZE/(VAL_SIZE+TEST_SIZE), stratify=True, seed=SEED
        )
        summary.summarize(
            summary.xy_summary(
                x_train, y_train, unique_groups=np.unique(groups_train), title="Train",
                x_label="x_train shape", y_label="y_train shape", groups_label="groups_train",
            ),
            summary.xy_summary(
                x_val, y_val, unique_groups=np.unique(groups_val), title="Val",
                x_label="x_val shape", y_label="y_val shape", groups_label="groups_val",
            ),
            summary.xy_summary(
                x_test, y_test, unique_groups=np.unique(groups_test), title="Test",
                x_label="x_test shape", y_label="y_test shape", groups_label="groups_test",
            ),
            filepath=summary_file
        )
        # Kfold generator
        cv_generator = models.cv_object(
            n_split=CV, groups=groups_train, stratify=True, seed=SEED
        )
        # Hyperparameters search
        search = models.random_forest_search(
            x=x_train, y=y_train.ravel(), groups=groups_train,
            n_split=CV, stratify=True, seed=SEED, verbose=1,
            scoring=SCORING, n_iter=N_ITER, refit=FIT_WITH, n_jobs=None,
            class_weight=TARGETS_WEIGHTS, return_train_score=CV_TRAIN,
            cv_generator=cv_generator, random_state=SEED,
            param_criterion=hsearch_criterion,
            param_n_estimators=hsearch_n_estimators,
            param_max_features=hsearch_max_features,
            param_max_depths=hsearch_max_depths,
            param_min_s_split=hsearch_min_s_split,
            param_min_s_leaf=hsearch_min_s_leaf,
            param_bootstrap=hsearch_bootstrap,
        )
        # Save tested parameters
        idx_search = search.best_index_
        df_search = pd.DataFrame(search.cv_results_)
        df_search.to_csv(hsearch_file)
        ## In clearer format
        val_scores = {"model": [], "mean": [], "std": [], "rank": [], "score": []}
        train_scores = {"model": [], "mean": [], "std": [], "score": []}
        for idx, key in enumerate(SCORING.keys()):
            key_score = [key] * N_ITER
            key_model = [i for i in range(N_ITER)]
            val_scores["model"].extend(key_model)
            val_scores["mean"].extend(search.cv_results_[f"mean_test_{key}"])
            val_scores["std"].extend(search.cv_results_[f"std_test_{key}"])
            val_scores["score"].extend(key_score)
            val_scores["rank"].extend(search.cv_results_[f"rank_test_{key}"])
            if CV_TRAIN:
                train_scores["model"].extend(key_model)
                train_scores["mean"].extend(search.cv_results_[f"mean_train_{key}"])
                train_scores["std"].extend(search.cv_results_[f"std_train_{key}"])
                train_scores["score"].extend(key_score)
        df_train_scores = None
        df_val_scores = pd.DataFrame(val_scores)
        df_val_scores.to_csv(hsearch_val_file, index=False)
        display.display_cv_scores(df_val_scores, filepath=cv_scores_val_file, title="CV performances")
        if CV_TRAIN:
            df_train_scores = pd.DataFrame(train_scores)
            df_train_scores.to_csv(hsearch_train_file, index=False)
            display.display_cv_scores(df_train_scores, filepath=cv_scores_train_file, title="CV performances")

        # CV - Confusion matrix
        observed_val, predicted_val = models.evaluate_kmodel(
            x=x_train, y=y_train, estimator=search.best_estimator_,
            kfolder=cv_generator.split(x_train, y_train.ravel(), groups_train),
            return_train=False
        )
        display.display_confusion_matrix(
            observed=observed_val, predicted=predicted_val,
            labels=None, normalize="true", filepath=cv_cfmatrix_val_file
        )

        cv_perf_val = dict()
        cv_perf_train = dict() if CV_TRAIN else None
        for idx, key in enumerate(SCORING.keys()):
            cv_perf_val[key] = \
                f"mean={df_search[f'mean_test_{key}'].iloc[idx_search]:.3f} "u'\u00b1'f" {df_search[f'std_test_{key}'].iloc[idx_search]:.3f}"
            if CV_TRAIN:
                cv_perf_train[key] = \
                    f"mean={df_search[f'mean_train_{key}'].iloc[idx_search]:.3f} "u'\u00b1'f" {df_search[f'std_train_{key}'].iloc[idx_search]:.3f}"

        summary.summarize(
            summary.arg_summary(
                "Best parameters",
                "\n" +
                summary.mapped_summary(
                    search.best_params_,
                    map_sep="=", padding_left=4
                ),
                new_line=False
            ),
            summary.arg_summary(
                "Val", "\n" + summary.mapped_summary(cv_perf_val, map_sep="=", padding_left=4),
                new_line=False
            ),
            title="Results",
            filepath=summary_file
        )
        if CV_TRAIN:
            summary.arg_summary(
                "Train", "\n" + summary.mapped_summary(cv_perf_train, map_sep="=", padding_left=4),
                filepath=summary_file
            )

        # Feature Importance
        ## Mean Decrease Impurity
        df_mdi = pd.DataFrame(
            models.forest_mdi_importance(rf_estimator=search.best_estimator_, colnames=features_column)
        )
        df_mdi.to_csv(mdi_importance_file, index=False)
        display.display_mdi_importance(mdi_importance=df_mdi, filepath=mdi_plot_file)
        ## Permutation
        permutation = models.forest_permutation_importance(
            estimator=search.best_estimator_, x=x_train, y=y_train.ravel(),
            colnames=features_column, seed=SEED
        )
        df_permutation = pd.DataFrame({
            "importances_mean": permutation["importances_mean"],
             "importances_std": permutation["importances_std"],
             "colnames": permutation["colnames"]
        })
        df_permutation.to_csv(permut_importance_train_file, index=False)
        display.display_permutation_importance(df_permutation, filepath=permut_plot_train_file)
        ## Boruta
        boruta = models.forest_boruta_importance(
            estimator=search.best_estimator_, x=x_train, y=y_train.ravel(),
            colnames=features_column, n_trials=30
        )
        df_boruta = pd.DataFrame(boruta["feature_hit"])
        df_boruta.to_csv(boruta_importance_train_file, index=False)
        display.display_boruta_importance(
            boruta_importance=df_boruta, treshold=boruta.treshold,
            n_trials=boruta.n_trials, filepath=boruta_plot_train_file
        )
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

