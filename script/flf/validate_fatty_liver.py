import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sail_model_optimizer.visualizations.visualizations import (  # generate_feature_heatmap,
    calculate_results,
    generate_feature_wordcloud,
    load_model_from_file,
    plot_feature_venn,
    plot_roc_curves,
    visualize_pca,
)

os.environ["PATH_DIR_RESULTS"] = "/home/adam/sail-model-optimizer/script/flf/results"
path_dir_results = os.environ["PATH_DIR_RESULTS"]


df_input_train = pd.read_csv("/home/adam/sail-model-optimizer/script/flf/data/fatty_liver_input_train.csv")
df_input_test = pd.read_csv("/home/adam/sail-model-optimizer/script/flf/data/fatty_liver_input_test.csv")
df_output_train = pd.read_csv("/home/adam/sail-model-optimizer/script/flf/data/fatty_liver_output_train.csv")
df_output_test = pd.read_csv("/home/adam/sail-model-optimizer/script/flf/data/fatty_liver_output_test.csv")


# Logistic Regression
path_file_results_lr = os.path.join(path_dir_results, "logistic.json")
lr_model, features_lr = load_model_from_file(path_file_results_lr, LogisticRegression, df_input_train, df_output_train)

# XGBoost
path_file_results_xgb = os.path.join(path_dir_results, "xgboost.json")
xgb_model, features_xgb = load_model_from_file(
    path_file_results_xgb, xgb.XGBClassifier, df_input_train, df_output_train
)

# Bayesian Network
path_file_results_bayesian = os.path.join(path_dir_results, "bayesian_network.json")
bayesian_model, features_bayesian = load_model_from_file(
    path_file_results_bayesian, GaussianNB, df_input_train, df_output_train
)

array_input_test_xgb = df_input_test[features_xgb].to_numpy()
array_input_test_lr = df_input_test[features_lr].to_numpy()
array_input_test_bayesian = df_input_test[features_bayesian].to_numpy()

model_data = [
    (xgb_model, array_input_test_xgb, "XGBoost"),
    (lr_model, array_input_test_lr, "Logistic Regression"),
    (bayesian_model, array_input_test_bayesian, "Bayesian Network"),
]

results = calculate_results(model_data, df_output_test)

plot_roc_curves(results)
feature_lists = [features_lr, features_xgb, features_bayesian]
set_labels = ("Logistic", "XGB", "Bayesian")
plot_feature_venn(feature_lists, set_labels)
generate_feature_wordcloud(feature_lists)
# generate_feature_frequency_graph(feature_lists)
visualize_pca(df_input_train, df_output_train, feature_lists)
