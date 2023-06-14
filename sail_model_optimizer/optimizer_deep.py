from pandas import DataFrame

from sail_model_optimizer.optimizer_base import OptimizerBase


class OptimizerDeep(OptimizerBase):
    def __init__(
        self,
        run_evaluator,
        run_dict_builder,
        fold_count: int = 5,
        fold_fracton: float = 0.1,
        optimize_hyper_parameter_count: int = 6,
    ) -> None:
        super().__init__(
            run_evaluator,
            fold_count,
            fold_fracton,
            optimize_hyper_parameter_count,
        )
        self.run_dict_builder = run_dict_builder

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
            dict_run_new = self.optimize_hyper_parameter(df_input, df_output, dict_run_new)
            if dict_run_best["score"] < dict_run_new["score"]:
                dict_run_best = dict_run_new
                # for display only
                score = dict_run_new["score"]
                list_feature_selected = dict_run_new["list_feature_selected"]
                print(f"new best score: {score}")
                print(f"list_feauter {list_feature_selected}")
        return dict_run_best

    def optimize_model(
        self,
        df_input: DataFrame,
        df_output: DataFrame,
    ) -> dict:
        # initial run
        dict_run = self.run_dict_builder.build_run_dict(df_input, df_output)
        dict_run = self.run_evaluator.evaluate_run(df_input, df_output, df_input, df_output, dict_run)
        score_best = dict_run["score"]
        # iterative run
        has_improvement = True
        while has_improvement:
            has_improvement = False
            print(f"Starting iteration")
            print(f"new best score {score_best}")
            dict_run = self.optimize_feature_selection(df_input, df_output, dict_run)

            if score_best < dict_run["score"]:
                score_best = dict_run["score"]
                print(f"!!!!new best score {score_best}")
                has_improvement = True
        return dict_run
