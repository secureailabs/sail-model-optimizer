import json
import os
import random
import warnings

from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve


class RunEvaluatorLogisticRegression:
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
        dict_params = dict_run["dict_params_current"]
        array_input_train = df_input_train[list_feature_selected].to_numpy()
        array_output_train = df_output_train.to_numpy()
        array_input_test = df_input_test[list_feature_selected].to_numpy()
        array_output_test_true = df_output_test_true.to_numpy()

        # PCA stuff
        # pca = PCA(n_components=40, whiten=True)
        # array_input_train = pca.fit_transform(array_input_train)
        # array_input_test = pca.transform(array_input_test)

        warnings.filterwarnings("ignore")

        model = LogisticRegression()
        model.set_params(**dict_params)
        model.fit(array_input_train, array_output_train)

        array_output_test_pred = model.predict_proba(array_input_test)[:, 1]
        fpr, tpr, _ = roc_curve(array_output_test_true, array_output_test_pred)
        dict_run["score"] = float(auc(fpr, tpr))
        return dict_run


class RunDictBuilderLogisticRegression:
    def __init__(self) -> None:
        pass

    def build_run_dict(self, df_input: DataFrame, df_output: DataFrame) -> dict:
        dict_run = {}

        path_dir_results = os.environ["PATH_DIR_RESULTS"]
        path_file_results = os.path.join(path_dir_results, f"logistic.json")

        if os.path.exists(path_file_results):
            try:
                print("Reading dict_run logistic")
                with open(path_file_results, "r") as file_model:
                    dict_run = json.load(file_model)
            except IOError:
                print("Error reading the file!")
        else:
            print("New dict_run logistic")
            starting_portion = 0.5
            margin = 0.1
            num_elements = random.randint(
                int(len(df_input.columns) * (starting_portion - margin)),
                int(len(df_input.columns) * (starting_portion + margin)),
            )
            random_permutation = random.sample(list(df_input.columns), num_elements)
            dict_run["list_feature_selected"] = random_permutation
            dict_run["dict_params_current"] = {
                "C": 1.0,
                "max_iter": 100,
            }
            dict_run["dict_params_config"] = {
                "C": {
                    "type": "multiply",
                    "list_value": [0.9, 1.1],
                    "resolution": 0.1,
                },
                "max_iter": {
                    "type": "add",
                    "list_value": [-1, 1],
                    "resolution": 1,
                },
            }
            dict_run["score"] = 0
        return dict_run

    def save_dict_to_results(self, dict_run: dict):
        path_dir_results = os.environ["PATH_DIR_RESULTS"]
        path_file_results = os.path.join(path_dir_results, f"logistic.json")
        with open(path_file_results, "w") as file_model:
            json.dump(dict_run, file_model)


## TODO: add categorical features to search
