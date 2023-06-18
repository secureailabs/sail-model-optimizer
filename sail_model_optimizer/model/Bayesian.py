import json
import os
import random
import warnings

from pandas import DataFrame
from sklearn.metrics import auc, roc_curve
from sklearn.naive_bayes import GaussianNB


class RunEvaluatorBayesianNetwork:
    def __init__(self):
        pass

    def evaluate_run(
        self,
        df_input_train: DataFrame,
        df_output_train: DataFrame,
        df_input_test: DataFrame,
        df_output_test_true: DataFrame,
        dict_run: dict,
    ) -> dict:
        list_feature_selected = dict_run["list_feature_selected"]
        array_input_train = df_input_train[list_feature_selected].to_numpy()
        array_output_train = df_output_train.to_numpy()
        array_input_test = df_input_test[list_feature_selected].to_numpy()
        array_output_test_true = df_output_test_true.to_numpy()

        warnings.filterwarnings("ignore")

        model = GaussianNB()
        model.fit(array_input_train, array_output_train.ravel())

        array_output_test_pred = model.predict_proba(array_input_test)[:, 1]
        fpr, tpr, _ = roc_curve(array_output_test_true, array_output_test_pred)
        dict_run["score"] = float(auc(fpr, tpr))
        return dict_run


class RunDictBuilderBayesianNetwork:
    def __init__(self) -> None:
        pass

    def build_run_dict(self, df_input: DataFrame, df_output: DataFrame) -> dict:
        dict_run = {}

        path_dir_results = os.environ["PATH_DIR_RESULTS"]
        path_file_results = os.path.join(path_dir_results, f"bayesian_network.json")

        if os.path.exists(path_file_results):
            try:
                print("Reading dict_run Bayesian network")
                with open(path_file_results, "r") as file_model:
                    dict_run = json.load(file_model)
            except IOError:
                print("Error reading the file!")
        else:
            print("New dict_run Bayesian network")
            starting_portion = 0.5
            margin = 0.1
            num_elements = random.randint(
                int(len(df_input.columns) * (starting_portion - margin)),
                int(len(df_input.columns) * (starting_portion + margin)),
            )
            random_permutation = random.sample(list(df_input.columns), num_elements)
            dict_run["list_feature_selected"] = random_permutation
            dict_run["dict_params_current"] = {}
            dict_run["score"] = 0

        return dict_run

    def save_dict_to_results(self, dict_run: dict):
        path_dir_results = os.environ["PATH_DIR_RESULTS"]
        path_file_results = os.path.join(path_dir_results, f"bayesian_network.json")
        with open(path_file_results, "w") as file_model:
            json.dump(dict_run, file_model)
