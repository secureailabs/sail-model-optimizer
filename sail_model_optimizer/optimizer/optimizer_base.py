import json
import os
from copy import deepcopy
from hashlib import sha256
from itertools import permutations
from typing import List

from pandas import DataFrame
from sklearn.model_selection import train_test_split


class OptimizerBase:
    def __init__(
        self, run_evaluator, fold_count: int = 5, fold_fracton=0.1, optimize_hyper_parameter_count: int = 6
    ) -> None:
        self.run_evaluator = run_evaluator
        self.fold_count = fold_count
        self.fold_fracton = fold_fracton
        self.optimize_hyper_parameter_count = optimize_hyper_parameter_count

    def wipe(self):
        path_dir_model = os.environ["PATH_DIR_RUN"]
        for file_model in os.listdir(path_dir_model):
            os.remove(os.path.join(path_dir_model, file_model))

    def hash_dict_sha256(
        self,
        dict_model: dict,
    ) -> str:

        json_string = json.dumps(dict_model, sort_keys=True)
        return sha256(json_string.encode("utf-8")).hexdigest()

    def evaluate_run_fold(
        self,
        df_input: DataFrame,
        df_output: DataFrame,
        dict_run: dict,
    ) -> dict:
        path_dir_model = os.environ["PATH_DIR_RUN"]
        path_file_model = os.path.join(path_dir_model, f"{self.hash_dict_sha256(dict_run)}.json")
        if os.path.exists(path_file_model):
            with open(path_file_model, "r") as file_model:
                dict_run = json.load(file_model)
        else:
            score = 0
            for i in range(self.fold_count):
                df_input_train, df_input_test, df_output_train, df_output_test = train_test_split(
                    df_input, df_output, test_size=self.fold_fracton, random_state=i
                )
                dict_run = self.run_evaluator.evaluate_run(
                    df_input_train, df_output_train, df_input_test, df_output_test, dict_run
                )
                score += dict_run["score"]
            dict_run["score"] = score / self.fold_count
            # TODO add score varriance here
            with open(path_file_model, "w") as file_model:
                json.dump(dict_run, file_model)
        return dict_run

    def build_grid(
        self,
        dict_params_initial: dict,
        dict_params_config: dict,
    ) -> dict:
        param_grid = {}
        for name_param, value_param in dict_params_initial.items():
            if name_param in dict_params_config:
                type_param = dict_params_config[name_param]["type"]
                if type_param == "add":
                    param_grid[name_param] = [
                        value_param + value for value in dict_params_config[name_param]["list_value"]
                    ]
                elif type_param == "multiply":
                    param_grid[name_param] = [
                        value_param * value for value in dict_params_config[name_param]["list_value"]
                    ]
                ## TODO: add categorical features to search space
            else:
                param_grid[name_param] = [value_param]
        return param_grid

    def mutate_pramam(self, dict_run_base: dict, name_param: dict):
        param_grid = self.build_grid(dict_run_base["dict_params_current"], dict_run_base["dict_params_config"])
        list_dict_run = []
        for value in param_grid[name_param]:
            dict_run_new = deepcopy(dict_run_base)
            dict_run_new["dict_params_current"][name_param] = value
            list_dict_run.append(dict_run_new)
        return list_dict_run

    def optimize_hyper_parameter(
        self,
        df_input: DataFrame,
        df_output: DataFrame,
        dict_run_best: dict,
    ) -> dict:
        print("start optimize parameter", flush=True)
        for i in range(self.optimize_hyper_parameter_count):
            print(f"iteration {i}", flush=True)
            for param_name in dict_run_best["dict_params_config"]:
                # print(f"optimize {param_name}", flush=True)
                list_dict_run = self.mutate_pramam(dict_run_best, param_name)
                for dict_run_new in list_dict_run:
                    param_value = dict_run_new["dict_params_current"][param_name]
                    # print(f"trying {param_value}", flush=True)
                    dict_run_new = self.evaluate_run_fold(df_input, df_output, dict_run_new)
                    # print(dict_run_best["score"])
                    # print(dict_run_new["score"])
                    if dict_run_best["score"] < dict_run_new["score"]:
                        dict_run_best = dict_run_new
                        # for display only
                        score = dict_run_new["score"]
                        param_value = dict_run_new["dict_params_current"][param_name]
                        print(f"new best score: {score}")
                        print(f"parameter {param_name} changed to: {param_value}")
        return dict_run_best

    def mutate_feature_selection(self, dict_run: dict, list_all_features: List[str]):
        list_dict_run = []
        for name_feature in list_all_features:
            dict_run_new = deepcopy(dict_run)
            if name_feature in dict_run_new["list_feature_selected"]:
                # print(f"removing {name_feature}")
                dict_run_new["list_feature_selected"].remove(name_feature)
            else:
                # print(f"adding {name_feature}")
                dict_run_new["list_feature_selected"].append(name_feature)

            # print("feature_count", len(dict_run_new["list_feature_selected"]))
            list_dict_run.append(dict_run_new)
        return list_dict_run

    def optimize_feature_selection(
        self,
        df_input: DataFrame,
        df_output: DataFrame,
        dict_run_best: dict,
    ) -> dict:
        print("optimize_featureset", flush=True)
        list_dict_run = self.mutate_feature_selection(dict_run_best, list(df_input.columns))
        for dict_run_new in list_dict_run:
            # print(f"trying {param_value}", flush=True)
            dict_run_new = self.evaluate_run_fold(df_input, df_output, dict_run_new)
            if dict_run_best["score"] < dict_run_new["score"]:
                dict_run_best = dict_run_new
                # for display only
                score = dict_run_new["score"]
                list_feature_selected = dict_run_new["list_feature_selected"]
                # print(f"new best score: {score}")
                # print(f"list_feauter {list_feature_selected}")
        return dict_run_best

    def optimize_model(
        self,
        df_input: DataFrame,
        df_output: DataFrame,
    ) -> dict:
        raise NotImplementedError()
