import json
import os

import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from sail_model_optimizer.model.LogReg import RunDictBuilderLogisticRegression, RunEvaluatorLogisticRegression
from sail_model_optimizer.model.XGBoost import RunDictBuilderXGBoost, RunEvaluatorXGBoost
from sail_model_optimizer.optimizer.optimizer_genetic import OptimizerGenetic

os.environ["PATH_DIR_RESULTS"] = "/home/adam/sail-model-optimizer/script/flf/results"
os.environ["PATH_DIR_RUN"] = "/home/adam/runs"


df_output = pd.read_csv("script/fatty_liver_output.csv")
df_input = pd.read_csv("script/fatty_liver_input.csv")

df_input_train, df_input_test, df_output_train, df_output_test = train_test_split(
    df_input,
    df_output,
    test_size=0.1,
    random_state=42,
)


# def run_optimizer(optimizer, df_input_train, df_output_train, df_input_test, df_output_test):
#     dict_run = optimizer.optimize_model(df_input_train, df_output_train)

#     path_dir_model = os.environ["PATH_DIR_RUN"]
#     path_file_model = os.path.join(path_dir_model, f"best.json")
#     with open(path_file_model, "r") as file_model:
#         dict_run = json.load(file_model)
#     print("training score")
#     print(dict_run["score"])
#     print(dict_run["list_feature_selected"])

#     dict_run_old = run_dict_builder.build_run_dict(df_input, df_output)

#     dict_run = run_evaluator.evaluate_run(df_input_train, df_output_train, df_input_test, df_output_test, dict_run)
#     print("test score")
#     print(dict_run["score"])
#     print(dict_run["dict_params_current"])

#     if dict_run["score"] > dict_run_old["score"]:

#         optimizer.run_dict_builder.save_dict_to_results(dict_run)
#         print("New best against test set")
#         # with open("script/best_models/logistic.json", "w") as file_model:
#         #     json.dump(dict_run, file_model)


# Logistic Regression model
# run_evaluator = RunEvaluatorLogisticRegression()
# run_dict_builder = RunDictBuilderLogisticRegression()
# optimizer = OptimizerGenetic(
#     run_evaluator,
#     run_dict_builder,
# )
# optimizer.run(df_input_train, df_output_train, df_input_test, df_output_test)

# XGBoost model
evaluators = [RunEvaluatorLogisticRegression(), RunEvaluatorXGBoost()]
dict_builders = [RunDictBuilderLogisticRegression(), RunDictBuilderXGBoost()]

for evaluator, builder in zip(evaluators, dict_builders):
    optimizer = OptimizerGenetic(
        evaluator,
        builder,
    )
    optimizer.run(df_input_train, df_output_train, df_input_test, df_output_test)

# XGBoost model

# run_evaluator = RunEvaluatorXGBoost()
# run_dict_builder = RunDictBuilderXGBoost()

# optimizer = OptimizerGenetic(
#     run_evaluator,
#     run_dict_builder,
# )
# dict_run = optimizer.optimize_model(df_input_train, df_output_train)
# path_dir_model = os.environ["PATH_DIR_RUN"]
# path_file_model = os.path.join(path_dir_model, f"best.json")

# with open(path_file_model, "r") as file_model:
#     dict_run = json.load(file_model)
# print("training score")
# print(dict_run["score"])
# print(dict_run["list_feature_selected"])

# if os.path.exists("script/best_models/xgboost.json"):
#     try:
#         with open("script/best_models/xgboost.json", "r") as file_model:
#             dict_run_old = json.load(file_model)
#     except IOError:
#         print("Error reading the file!")
# else:
#     print("New dict_run_old xgboost")
#     dict_run_old = RunDictBuilderXGBoost().build_run_dict(df_input, df_output)
#     dict_run_old = run_evaluator.evaluate_run(
#         df_input_train, df_output_train, df_input_test, df_output_test, dict_run_old
#     )

# dict_run = run_evaluator.evaluate_run(df_input_train, df_output_train, df_input_test, df_output_test, dict_run)
# print("test score")
# print(dict_run["score"])
# print(dict_run["dict_params_current"])

# if dict_run["score"] > dict_run_old["score"]:
#     print("New best against test set")
#     with open("script/best_models/xgboost.json", "w") as file_model:
#         json.dump(dict_run, file_model)
# else:
#     print("!!!NO improvement against test set!!!")
