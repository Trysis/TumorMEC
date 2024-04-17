import os

import src.utils.constantes as cst
import src.utils.summary as summary
import src.utils.auxiliary as auxiliary
import src.processing.dataloader as load
import src.models.scorer as scorer
import src.models.models as models
import src.models.display_results as display

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Generic attributes
SEED = 42
TEST_SIZE = 0.3

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

# Attributes for 2nd DataLoader to compare
MASK_CONDITION_2 = MASK_CONDITION
MASK_TYPE_2 = [cst.LY6]
MASK_TUMOR_2 = MASK_TUMOR
MASK_DENSITY_2 = MASK_DENSITY

REMOVE_NONE_2 = REMOVE_NONE
REPLACE_ABERRANT_2 = REPLACE_ABERRANT

# Attributes specifying features, target and sample
FEATURES = {"loc-fract": cst.x_fiber_columns}
TARGETS = [load.enrich_2_cmask] #[load.plus_cmask, load.enrich_cmask, load.enrich_2_cmask]
TARGETS_COLNAMES = [target_col(return_key=True) for target_col in TARGETS]

SAMPLE_GROUP = []  # TODO : Replace by None
REMOVE_SAMPLE = None
TRAIN_NEW_MODEL = False

# Training regimen
CV = 8
LEAVE_ONE_OUT = False  # If True, CV is not used
N_PROCESS = max(CV, 1)  # Multi-threading
CV_TRAIN = True
TRAIN = True
SCORING = {
    "accuracy": scorer.accuracy_score(to_scorer=True),
    "balanced_accuracy": scorer.balanced_accuracy_score(to_scorer=True),
    "precision": scorer.precision_score(to_scorer=True),
    "recall": scorer.recall_score(to_scorer=True),
    "auc": scorer.roc_auc_score(to_scorer=True),
    "mcc": scorer.matthews_corrcoef(to_scorer=True),
    "f1": scorer.f1_score(to_scorer=True),
}
SCORING_base = {
    "accuracy": scorer.accuracy_score(to_scorer=False),
    "balanced_accuracy": scorer.balanced_accuracy_score(to_scorer=False),
    "precision": scorer.precision_score(to_scorer=False),
    "recall": scorer.recall_score(to_scorer=False),
    "auc": scorer.roc_auc_score(to_scorer=False),
    "mcc": scorer.matthews_corrcoef(to_scorer=False),
    "f1": scorer.f1_score(to_scorer=False),
}
FIT_WITH = "f1"

# Importances attributes
N_PERM = 30
N_BORUTA = None

## Process
SAMPLE_GROUP = None if SAMPLE_GROUP == [] else SAMPLE_GROUP
CV = 1 if LEAVE_ONE_OUT else CV

if __name__ == "__main__":
    # Load data 1 to retrieve model
    loader_1 = load.DataLoader(
        data_dir=DATA_DIR,
        mask_condition=MASK_CONDITION,
        mask_type=MASK_TYPE,
        mask_tumor=MASK_TUMOR,
        mask_fiber=MASK_DENSITY,
        replace_aberrant=REPLACE_ABERRANT,
        aberrant_columns=cst.aberrant_columns,
        remove_none=REMOVE_NONE,
    )
    # Load data 2 to test model on this dataset
    loader_2 = load.DataLoader(
        data_dir=DATA_DIR,
        mask_condition=MASK_CONDITION_2,
        mask_type=MASK_TYPE_2,
        mask_tumor=MASK_TUMOR_2,
        mask_fiber=MASK_DENSITY_2,
        replace_aberrant=REPLACE_ABERRANT_2,
        aberrant_columns=cst.aberrant_columns,
        remove_none=REMOVE_NONE_2,
    )
    dataframe_1 = loader_1.load_data(
        targets=TARGETS,
        type=cst.data_type,
        save=False,
        force_default=False,
        remove_sample=REMOVE_SAMPLE,
    )

    dataframe_2 = loader_2.load_data(
        targets=TARGETS,
        type=cst.data_type,
        save=False,
        force_default=False,
        remove_sample=REMOVE_SAMPLE,
    )
    filename_1 = loader_1.filename_from_mask()
    rootname_1, ext = os.path.splitext(filename_1)
    rootname_1 = "LEAVE-ONE-OUT_" + rootname_1 if LEAVE_ONE_OUT else rootname_1
    rootname_1 = "UNGROUP_" + rootname_1 if SAMPLE_GROUP is None else rootname_1

    filename = loader_2.filename_from_mask()
    rootname, ext = os.path.splitext(filename)
    rootname = "LEAVE-ONE-OUT_" + rootname if LEAVE_ONE_OUT else rootname
    rootname = "UNGROUP_" + rootname if SAMPLE_GROUP is None else rootname
    # Checks if we compare WT with KI, or CD3 with LY6
    rootname = f"COND=WT-KI_{rootname}" if MASK_CONDITION != MASK_CONDITION_2 else \
        f"TYPE=CD3-LY6_{rootname}" if MASK_TYPE != MASK_TYPE_2 else ""

    print(rootname_1)
    print(rootname)
    # Summary either for new model, or an already existant one
    summary_name = "summary.txt" if TRAIN_NEW_MODEL else "summary_estimator.txt"
    param_name = "search_param.csv" if TRAIN_NEW_MODEL else "estimator_param.csv"
    param_reduced_name = "search_param-reduced_val.csv" if TRAIN_NEW_MODEL else "estimator_param_val.csv"
    model_name = "best_estimator.joblib" if TRAIN_NEW_MODEL else "estimator.joblib"
    # Define X and Y
    for target_column in TARGETS_COLNAMES:
        for key, features_column in FEATURES.items():
            # Saving output
            loader_name_1 = f"{rootname_1}_{key}_{target_column}"  # MAIN DIR
            loader_dir_1 = auxiliary.create_dir(os.path.join(OUTPUT_DIR, loader_name_1), add_suffix=False)  #
            hsearch_file_1 = os.path.join(loader_dir_1, "search_param.csv")  #

            loader_name = f"{rootname}_{key}_{target_column}"  # MAIN DIR
            loader_dir = auxiliary.create_dir(os.path.join(OUTPUT_DIR, loader_name), add_suffix=False)  #
            # dir
            model_dir = auxiliary.create_dir(os.path.join(loader_dir, "main"))
            train_test_dir = auxiliary.create_dir(os.path.join(loader_dir, "train_test"))
            train_test_plot_dir = auxiliary.create_dir(os.path.join(train_test_dir, "plots"))
            train_test_n_dir = auxiliary.create_dir(os.path.join(loader_dir, "n_models"))
            train_test_n_plot_dir = auxiliary.create_dir(os.path.join(train_test_n_dir, "plots"))
            train_test_n_i_dir = os.path.join(train_test_n_dir, "model_{i}")  # use with str.format
            train_test_n_i_plot_dir = os.path.join(train_test_n_i_dir, "plots")  # use with str.format
            # files
            summary_file = os.path.join(loader_dir, summary_name)
            hsearch_file = os.path.join(loader_dir, "search_param.csv")  #
            hsearch_after_file = os.path.join(loader_dir, param_name)
            model_file = os.path.join(model_dir, model_name)

            scores_train_file = os.path.join(train_test_dir, "scores_train.csv")
            scores_test_file = os.path.join(train_test_dir, "scores_test.csv")
            train_test_score_plot_file = os.path.join(train_test_plot_dir, "train-test_score.png")
            cfmatrix_plot_train_file = os.path.join(train_test_plot_dir, "cfmatrix_train.png")
            cfmatrix_plot_test_file = os.path.join(train_test_plot_dir, "cfmatrix_test.png")
            
            mdi_importance_file = os.path.join(train_test_dir, "mean-decrease-impurity.csv")
            raw_permut_importance_train_file = os.path.join(train_test_dir, "raw_permutation_train.csv")
            raw_permut_importance_test_file = os.path.join(train_test_dir, "raw_permutation_test.csv")
            permut_importance_train_file = os.path.join(train_test_dir, "permutation_train.csv")
            permut_importance_test_file = os.path.join(train_test_dir, "permutation_test.csv")
            boruta_importance_train_file = os.path.join(train_test_dir, "boruta_train.csv")
            boruta_importance_test_file = os.path.join(train_test_dir, "boruta_test.csv")
            mdi_plot_file = os.path.join(train_test_plot_dir, "mean-decrease-impurity.png")
            permut_plot_train_file = os.path.join(train_test_plot_dir, "permutation_train.png")
            permut_plot_train_boxplot_file = os.path.join(train_test_plot_dir, "permutation_boxplot_train.png")
            permut_plot_train_violin_file = os.path.join(train_test_plot_dir, "permutation_violin_train.png")
            permut_plot_test_file = os.path.join(train_test_plot_dir, "permutation_test.png")
            permut_plot_test_boxplot_file = os.path.join(train_test_plot_dir, "permutation_boxplot_test.png")
            permut_plot_test_violin_file = os.path.join(train_test_plot_dir, "permutation_violin_test.png")
            boruta_plot_train_file = os.path.join(train_test_plot_dir, "boruta_train.png")
            boruta_plot_test_file = os.path.join(train_test_plot_dir, "boruta_test.png")
            shap_plot_train_file = os.path.join(train_test_plot_dir, "shap_train.png")
            shap_plot_test_file = os.path.join(train_test_plot_dir, "shap_test.png")
            # For the N models (with N=Number of folds)
            model_i_file = os.path.join(model_dir, "estimator_{i}.joblib")
            n_scores_test_file = os.path.join(train_test_n_dir, "n_scores_test.csv")
            mdi_i_importance_file = "model_{i}_mean-decrease-impurity.csv"  # to join
            mdi_i_plot_file = "model_{i}_mean-decrease-impurity.png"
            raw_i_permut_importance_train_file = "model_{i}_raw_permutation_train.csv"
            raw_i_permut_importance_test_file = "model_{i}_raw_permutation_test.csv"
            permut_i_importance_train_file = "model_{i}_permutation_train.csv"
            permut_i_importance_test_file = "model_{i}_permutation_test.csv"
            permut_i_plot_train_file = "model_{i}_permutation_train.png"
            permut_i_plot_test_file = "model_{i}_permutation_test.png"
            n_confusion_test_file = os.path.join(train_test_n_dir, "n_confusion_test.csv")
            n_confusion_train_file = os.path.join(train_test_n_dir, "n_confusion_train.csv")
            n_mdi_file = os.path.join(train_test_n_dir, "n_mdi.csv")
            n_permut_train_file = os.path.join(train_test_n_dir, "n_permutation_train.csv")
            n_permut_test_file = os.path.join(train_test_n_dir, "n_permutation_test.csv")
            mean_cfmatrix_plot_train_file = os.path.join(train_test_n_plot_dir, "cfmatrix_train.png")
            mean_cfmatrix_plot_test_file = os.path.join(train_test_n_plot_dir, "cfmatrix_test.png")

            summary.summarize(
                summary.mapped_summary({
                    "Condition": [c.name for c in MASK_CONDITION],
                    "Type": [t.name for t in MASK_TYPE],
                    "Tumor": [t.name for t in MASK_TUMOR],
                    "Fiber": [d.name for d in MASK_DENSITY],
                    "Remove none": REMOVE_NONE,
                    "Replace aberrant": REPLACE_ABERRANT,
                    "NJOBS (Multi-threading)": N_PROCESS,
                    "SEED": SEED,
                }, map_sep=":"),
                summary.arg_summary(
                    "Importance parameters",
                    "\n" +
                    summary.mapped_summary({
                        "N-Permutation shuffle": N_PERM,
                        "N-Boruta trials": N_BORUTA,
                    }, map_sep="=", padding_left=4),
                    new_line=False
                ),
                title="Parameters",
                filepath=summary_file, mode="w"
            )

            summary.summarize(
                summary.mapped_summary({
                    "MODEL": models.ESTIMATOR,
                    "TEST RATIO": TEST_SIZE,
                    "GROUPS": True if SAMPLE_GROUP else False,
                    "Cross-validation N-Folds": "Leave one out" if LEAVE_ONE_OUT else CV,
                }, map_sep=":"),
                summary.arg_summary(
                    "Scoring",
                    "\n" + summary.mapped_summary(SCORING, padding_left=4),
                    new_line=False
                ),
                subtitle="Training regiment",
                filepath=summary_file
            )

            # Features and Target(s)
            x, y, groups = models.split_xy(
                df=dataframe_1, x_columns=features_column, y_columns=target_column, groups=SAMPLE_GROUP
            )
            mapped_groups = None
            if SAMPLE_GROUP:
                label_groups = pd.DataFrame(dataframe_2[SAMPLE_GROUP].agg(';'.join, axis=1), columns=["label"])
                df_mapped_groups = label_groups.assign(groups=groups).drop_duplicates()    
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
            x_train, x_test, y_train, y_test, groups_train, groups_test = models.split_data(
                x, y, groups=groups, n_splits=1, test_size=TEST_SIZE, stratify=True, seed=SEED
            )
            u_groups_train = np.unique(groups_train) if groups_train is not None else None
            u_groups_test = np.unique(groups_test) if groups_test is not None else None
            if not LEAVE_ONE_OUT:
                summary.summarize(
                    summary.xy_summary(
                        x_train, y_train, unique_groups=u_groups_train, title="Train",
                        x_label="x_train shape", y_label="y_train shape", groups_label="groups_train",
                    ),
                    summary.xy_summary(
                        x_test, y_test, unique_groups=u_groups_test, title="Test",
                        x_label="x_test shape", y_label="y_test shape", groups_label="groups_test",
                    ),
                    filepath=summary_file
                )

            ####### FIT Model to First DATA
            # Kfold generator
            cv_generator = models.cv_object(
                n_split=CV, groups=groups_train, stratify=True, seed=SEED
            )
            # Model
            estimator = None  # model trained on train
            estimator_list = []  # models trained on train cv
            estimator_param = dict()  # model parameters
            # Score of model on each cv split
            df_scores = None
            idx_scores = None  # best model row index
            # Hyperparameters search
            search_results = None
            if auxiliary.isfile(hsearch_file_1):
                search_results = pd.read_csv(hsearch_file_1)
                search_param = search_results.loc[search_results[f"rank_test_{FIT_WITH}"] == 1]["params"]
                estimator_param = eval(search_param.values[0])
            elif isinstance(TRAIN_NEW_MODEL, str): # Search for other file ? or apply randomsearch ?
                if auxiliary.isfile(TRAIN_NEW_MODEL):
                    search_results = pd.read_csv(TRAIN_NEW_MODEL)
            else:
                raise Exception("Param File not found, change TRAIN_NEW_MODEL to False")

            # Model from already searched parameters
            estimator = RandomForestClassifier(**estimator_param, random_state=SEED)
            estimator.fit(x_train, y_train.ravel())
            idx_scores = 0
        
            # List of model trained on each Train CV
            dict_scores = dict()
            for i, (train_index, val_index) in enumerate(cv_generator.split(x_train, y_train, groups_train)):
                model_i = RandomForestClassifier(**estimator_param, random_state=SEED)
                model_i.fit(x[train_index, :], y[train_index].ravel())
                estimator_list.append(model_i)
                # Save model {i}
                joblib.dump(model_i, model_i_file.format(i=i))
                if not TRAIN_NEW_MODEL:
                    dict_scores_i = models.scorer_model(
                        model_i, x[val_index, :], y[val_index], SCORING_base,
                        prefix=f"split{i}_test_"
                    )
                    dict_scores = {**dict_scores, **dict_scores_i}
                    if CV_TRAIN:
                        dict_scores_i_train = models.scorer_model(
                            model_i, x[train_index, :], y[train_index], SCORING_base,
                            prefix=f"split{i}_train_"
                        )
                        dict_scores = {**dict_scores, **dict_scores_i_train}

            ################## NOW evaluate on seconde dataset
            x, y, groups = models.split_xy(
                df=dataframe_2, x_columns=features_column, y_columns=target_column, groups=SAMPLE_GROUP
            )
            x_train, x_test, y_train, y_test, groups_train, groups_test = models.split_data(
                x, y, groups=groups, n_splits=1, test_size=TEST_SIZE, stratify=True, seed=SEED
            )
            if not LEAVE_ONE_OUT:
                # Train, Test - Score
                ## Train
                train_scores = models.scorer_model(
                    estimator=estimator,
                    x=x_train, y=y_train, scorer=SCORING_base
                )

                summary.arg_summary(
                    "Train", "\n" + summary.mapped_summary(
                        train_scores, map_sep="=", padding_left=4
                    ), filepath=summary_file
                )

                df_train_scores = pd.DataFrame(train_scores).T
                df_train_scores.to_csv(scores_train_file)
                ## Test
                test_scores = models.scorer_model(
                    estimator=estimator,
                    x=x_test, y=y_test, scorer=SCORING_base
                )
                summary.arg_summary(
                    "Test", "\n" + summary.mapped_summary(
                        test_scores, map_sep="=", padding_left=4
                    ), filepath=summary_file
                )
                df_test_scores = pd.DataFrame(test_scores).T
                df_test_scores.to_csv(scores_test_file)

                # N models - Calculate score on test
                n_test_scores_dict = dict()
                for i, model_i in enumerate(estimator_list):
                    scores_test_i = models.scorer_model(
                        estimator=model_i,
                        x=x_test, y=y_test, scorer=SCORING_base,
                        prefix=f"model_{i}"
                    )
                    n_test_scores_dict = {**n_test_scores_dict, **scores_test_i}
                # Save to csv
                pd.DataFrame(n_test_scores_dict).to_csv(n_scores_test_file)

                # Train, Test - Score
                display.display_train_test_scores(
                    train_scores=train_scores,
                    test_scores=test_scores,
                    title="Score", filepath=train_test_score_plot_file
                )
                # Train, Test - Confusion matrix
                if TRAIN:
                    observed_train, predicted_train = models.predict_model(x_train, y_train, estimator)
                    display.display_confusion_matrix(
                        observed=observed_train, predicted=predicted_train, cmap="Reds",
                        labels=None, filepath=cfmatrix_plot_train_file, title=f"train: {target_column}"
                    )
                observed_test, predicted_test = models.predict_model(x_test, y_test, estimator)
                display.display_confusion_matrix(
                    observed=observed_test, predicted=predicted_test, cmap="Greens",
                    labels=None, filepath=cfmatrix_plot_test_file, title=f"test: {target_column}"
                )

                # Apply it with list of model
                ## Confusion matrix
                obs_pred_test_list = list()
                confusion_test_dict = dict()
                obs_pred_train_list = list()
                confusion_train_dict = dict()
                for i, model_i in enumerate(estimator_list):
                    observed_test, predicted_i_test = models.predict_model(x_test, y_test, model_i)
                    confusion_test_i = models.get_tn_fp_fn_tp(
                        observed_test, predicted_i_test,
                        prefix=f"model_{i}"
                    )
                    obs_pred_test_list.append([observed_test, predicted_i_test])
                    confusion_test_dict = {**confusion_test_dict, **confusion_test_i}
                    if TRAIN:
                        observed_train, predicted_i_train = models.predict_model(x_train, y_train, model_i)
                        confusion_train_i = models.get_tn_fp_fn_tp(
                            observed_train, predicted_i_train,
                            prefix=f"model_{i}"
                        )
                        obs_pred_train_list.append([observed_train, predicted_i_train])
                        confusion_train_dict = {**confusion_train_dict, **confusion_train_i}
                # Save tn, fp, fn, tp
                pd.DataFrame(confusion_test_dict).to_csv(n_confusion_test_file, index=False)
                display.display_mean_confusion_matrix(
                    obs_pred_test_list, cmap="Greens", labels=None,
                    filepath=mean_cfmatrix_plot_test_file, title=f"mean test performance"
                )
                if TRAIN:
                    pd.DataFrame(confusion_train_dict).to_csv(n_confusion_train_file, index=False)
                    display.display_mean_confusion_matrix(
                        obs_pred_train_list, cmap="Reds", labels=None,
                        filepath=mean_cfmatrix_plot_train_file, title=f"mean train performance"
                    )