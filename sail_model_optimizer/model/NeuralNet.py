import json
import os
import random
import warnings

import numpy as np
from pandas import DataFrame
from sklearn.metrics import auc, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


class RunEvaluatorNeuralNetwork:
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

        warnings.filterwarnings("ignore")
        # Set the random seed
        random_seed = 42
        np.random.seed(random_seed)

        # Create and train the Neural Network model
        model = MLPClassifier(random_state=random_seed)
        model.set_params(**dict_params)
        model.fit(array_input_train, array_output_train)

        # Normalize the input data
        scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
        array_input_train = scaler.fit_transform(array_input_train)
        array_input_test = scaler.transform(array_input_test)

        array_output_test_pred = model.predict_proba(array_input_test)[:, 1]
        array_output_test_pred = np.round(array_output_test_pred, 1)  # Round to two decimal places
        fpr, tpr, _ = roc_curve(array_output_test_true, array_output_test_pred)
        dict_run["score"] = float(auc(fpr, tpr))

        # # debugging wonky score
        # if dict_run["score"] > 0.99:
        #     print("orint predictions: " + str(array_output_test_pred))
        #     print("print value counts of output test true" + str(df_output_test_true.value_counts()))
        #     print("debugging wonky score: " + str(dict_run["score"]))
        #     print("The AUC" + str(auc(fpr, tpr)))
        #     print("the size of the test set is: " + str(len(array_output_test_true)))
        #     print("TPR: " + str(tpr))
        #     print("FPR: " + str(fpr))
        #     for pred, true in zip(array_output_test_pred, array_output_test_true):
        #         if pred != true:
        #             print(f"pred: {pred}, true: {true}")

        return dict_run


class RunDictBuilderNeuralNetwork:
    def __init__(self) -> None:
        pass

    def build_run_dict(self, df_input: DataFrame, df_output: DataFrame) -> dict:
        dict_run = {}

        path_dir_results = os.environ["PATH_DIR_RESULTS"]
        path_file_results = os.path.join(path_dir_results, f"neural_network.json")

        if os.path.exists(path_file_results):
            try:
                print("Reading dict_run neural network")
                with open(path_file_results, "r") as file_model:
                    dict_run = json.load(file_model)
            except IOError:
                print("Error reading the file!")
        else:
            print("New dict_run neural network")
            starting_portion = 0.3
            margin = 0.05
            num_elements = random.randint(
                int(len(df_input.columns) * (starting_portion - margin)),
                int(len(df_input.columns) * (starting_portion + margin)),
            )
            random_permutation = random.sample(list(df_input.columns), num_elements)
            dict_run["list_feature_selected"] = random_permutation
            dict_run["dict_params_current"] = {
                "hidden_layer_sizes": 8,
                "alpha": 0.01,
                "solver": "adam",
                "activation": "relu",
                "batch_size": "auto",
                "learning_rate": "constant",
            }
            dict_run["dict_params_config"] = {
                "hidden_layer_sizes": {
                    "type": "add",
                    "list_value": [-1, 1],
                    "resolution": 1,
                },
                # "activation": {
                #     "type": "categorical",
                #     "list_value": ["relu", "tanh", "logistic"],
                # },
                # "solver": {
                #     "type": "categorical",
                #     "list_value": ["adam", "sgd"],
                # },
                "alpha": {
                    "type": "multiply",
                    "list_value": [0.9, 1.1],
                    "resolution": 0.0001,
                },
                # "batch_size": {
                #     "type": "categorical",
                #     "list_value": ["auto", 32, 64, 128],
                # },
                # "learning_rate": {
                #     "type": "categorical",
                #     "list_value": ["constant", "invscaling", "adaptive"],
                # },
            }
            dict_run["score"] = 0
        return dict_run

    def save_dict_to_results(self, dict_run: dict):
        path_dir_results = os.environ["PATH_DIR_RESULTS"]
        path_file_results = os.path.join(path_dir_results, f"neural_network.json")
        with open(path_file_results, "w") as file_model:
            json.dump(dict_run, file_model)
