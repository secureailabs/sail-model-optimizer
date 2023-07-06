import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from sail_model_optimizer.model.Bayesian import RunDictBuilderBayesianNetwork, RunEvaluatorBayesianNetwork
from sail_model_optimizer.model.LogReg import RunDictBuilderLogisticRegression, RunEvaluatorLogisticRegression
from sail_model_optimizer.model.NeuralNet import RunDictBuilderNeuralNetwork, RunEvaluatorNeuralNetwork
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

# save train test splits to file
df_input_train.to_csv("/home/adam/sail-model-optimizer/script/flf/data/fatty_liver_output_test.csv", index=False)
df_input_test.to_csv("/home/adam/sail-model-optimizer/script/flf/data/fatty_liver_output_test.csv", index=False)
df_output_train.to_csv("/home/adam/sail-model-optimizer/script/flf/data/fatty_liver_output_test.csv", index=False)
df_output_test.to_csv("/home/adam/sail-model-optimizer/script/flf/data/fatty_liver_output_test.csv", index=False)

evaluators = [
    RunEvaluatorLogisticRegression(),
    RunEvaluatorXGBoost(),
    RunEvaluatorNeuralNetwork(),
    RunEvaluatorBayesianNetwork(),
]
dict_builders = [
    RunDictBuilderLogisticRegression(),
    RunDictBuilderXGBoost(),
    RunDictBuilderNeuralNetwork(),
    RunDictBuilderBayesianNetwork(),
]

for evaluator, builder in zip(evaluators, dict_builders):
    optimizer = OptimizerGenetic(
        evaluator,
        builder,
    )
    optimizer.run(df_input_train, df_output_train, df_input_test, df_output_test)
