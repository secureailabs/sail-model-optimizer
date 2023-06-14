from pandas import DataFrame

from sail_model_optimizer.optimizer_base import OptimizerBase


class OptimizerFlip(OptimizerBase):
    def __init__(self, run_evaluator, run_dict_builder, fold_count=5, optimize_hyper_parameter_count: int = 6) -> None:
        super().__init__(run_evaluator, fold_count, optimize_hyper_parameter_count)
        self.run_dict_builder = run_dict_builder

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
            dict_run = self.optimize_hyper_parameter(df_input, df_output, dict_run)
            if score_best < dict_run["score"]:
                score_best = dict_run["score"]
                print(f"!!!!new best score {score_best}")
                has_improvement = True
        return dict_run
