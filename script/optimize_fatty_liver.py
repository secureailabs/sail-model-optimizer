
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sail_model_optimizer.optimizer_deep import OptimizerDeep
from sail_model_optimizer.optimizer_flip import OptimizerFlip


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
        dict_run["list_feature_selected"] = list(df_input.columns)
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


# df_output = pd.read_csv("fatty_liver_output.csv")
# df_input = pd.read_csv("fatty_liver_input.csv")

# df_input_train, df_input_test, df_output_train, df_output_test = train_test_split(
#     df_input,
#     df_output,
#     test_size=0.1,
#     random_state=43,
# )

# run_evaluator = RunEvaluatorXGBoost()
# run_dict_builder = RunDictBuilderXGBoost()


# # optimizer = OptimizerFlip(
# #     run_evaluator,
# #     run_dict_builder,
# # )
# # dict_run = optimizer.optimize_model(df_input, df_output)
# # print("training score")
# # print(dict_run["score"])
# # print(dict_run["list_feature_selected"])
# # dict_run = run_evaluator.evaluate_run(df_input_train, df_output_train, df_input_test, df_output_test, dict_run)
# # print("test score")
# # print(dict_run["score"])
# # print(dict_run["dict_params_current"])


# optimizer = OptimizerDeep(
#     run_evaluator,
#     run_dict_builder,
#     fold_count=4,
#     fold_fracton=0.1,
# )
# optimizer.wipe()
# dict_run = optimizer.optimize_model(df_input, df_output)
# print("training score")
# print(dict_run["score"])
# print(dict_run["list_feature_selected"])
# dict_run = run_evaluator.evaluate_run(df_input_train, df_output_train, df_input_test, df_output_test, dict_run)
# print("test score")
# print(dict_run["score"])
# print(dict_run["dict_params_current"])
