SEED = 42

OUTDIR = constantes.OUTDIR
FILEPATH = ""
CROSS_VAL = 10
PREDICT_CLASS = ""

TYPE = "cd3"
CONDITION = ["wt", "ki"]
SESSION = f"{TYPE}_{'-'.join(CONDITION)}_main" 
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

