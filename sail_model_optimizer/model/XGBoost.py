import random

from pandas import DataFrame
from sklearn.metrics import auc, roc_curve
from xgboost import XGBClassifier


class RunEvaluatorXGBoost:
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

        model = XGBClassifier()
        model.set_params(**dict_params)
        model.fit(array_input_train, array_output_train)

        array_output_test_pred = model.predict_proba(array_input_test)[:, 1]
        fpr, tpr, _ = roc_curve(array_output_test_true, array_output_test_pred)
        dict_run["score"] = float(auc(fpr, tpr))
        return dict_run


class RunDictBuilderXGBoost:
    def __init__(self) -> None:
        pass

    def build_run_dict(self, df_input: DataFrame, df_output: DataFrame) -> dict:
        dict_run = {}
        starting_portion = 0.6
        margin = 0.05
        num_elements = random.randint(
            int(len(df_input.columns) * (starting_portion - margin)),
            int(len(df_input.columns) * (starting_portion + margin)),
        )
        random_permutation = random.sample(list(df_input.columns), num_elements)
        dict_run["list_feature_selected"] = random_permutation
        dict_run["dict_params_current"] = {
            "min_child_weight": 5,
            "gamma": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "max_depth": 5,
            "scale_pos_weight": 0.5,
            "eta": 0.3,
        }

        dict_run["dict_params_config"] = {
            "min_child_weight": {
                "type": "add",
                "list_value": [-1, 1],
                "resolution": 1,
            },
            "gamma": {
                "type": "multiply",
                "list_value": [0.99, 1.01],
                "resolution": 0.01,
            },
            "subsample": {
                "type": "multiply",
                "list_value": [0.99, 1.01],
                "resolution": 0.01,
                "min_value": 0,
                "max_value": 1,
            },
            "colsample_bytree": {
                "type": "multiply",
                "list_value": [0.99, 1.01],
                "resolution": 0.01,
            },
            # "max_depth": {
            #     "type": "add",
            #     "list_value": [-1, 1],
            #     "resolution": 1,
            #     "min_value": 1.0,
            # },
            "scale_pos_weight": {
                "type": "multiply",
                "list_value": [0.99, 1.01],
                "resolution": 0.01,
            },
            "eta": {
                "type": "multiply",
                "list_value": [0.99, 1.01],
                "resolution": 0.01,
            },
        }
        dict_run["score"] = 0
        return dict_run
