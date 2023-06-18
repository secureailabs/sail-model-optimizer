import os

import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from sail_model_optimizer.model.XGBoost import RunDictBuilderXGBoost, RunEvaluatorXGBoost
from sail_model_optimizer.optimizer.optimizer_genetic import OptimizerGenetic

os.environ["PATH_DIR_RUN"] = "/home/adam/runs"

df_output = pd.read_csv("script/fatty_liver_output.csv")
df_input = pd.read_csv("script/fatty_liver_input.csv")

df_input_train, df_input_test, df_output_train, df_output_test = train_test_split(
    df_input,
    df_output,
    test_size=0.1,
    random_state=43,
)

run_evaluator = RunEvaluatorXGBoost()
run_dict_builder = RunDictBuilderXGBoost()

optimizer = OptimizerGenetic(
    run_evaluator,
    run_dict_builder,
)
dict_run = optimizer.optimize_model(df_input_train, df_output_train)
print("training score")
print(dict_run["score"])
print(dict_run["list_feature_selected"])
dict_run = run_evaluator.evaluate_run(df_input_train, df_output_train, df_input_test, df_output_test, dict_run)
print("test score")
print(dict_run["score"])
print(dict_run["dict_params_current"])
