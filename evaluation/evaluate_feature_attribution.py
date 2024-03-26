import pandas as pd
from utils.helper_functions import rank_based_on_column_per_query


def get_estimated_ground_truth_feature_importance(file):
    feature_attribute_df = pd.read_csv(file)
    return feature_attribute_df[
        ["query_number", "feature_number", "attribution_value", "std"]
    ]  # TODO why do we use the std here?


def eval_feature_attribution(
    attributes_to_evaluate,
    ground_truth_file_path="../results/feature_attributes/"
    + "attribution_values_for_evaluation.csv",
):
    attribution_df = pd.read_csv(attributes_to_evaluate, index_col=0)
    ground_truth_attributes = get_estimated_ground_truth_feature_importance(
        ground_truth_file_path
    )
    results_df = pd.DataFrame()

    # prepare dataframe by adding some helper columns.
    attribution_df = attribution_df.merge(
        ground_truth_attributes,
        on=["query_number", "feature_number"],
        suffixes=("_exp", "_gt"),
    )
    attribution_df["abs_difference"] = (
        (
            attribution_df["attribution_value_exp"]
            - attribution_df["attribution_value_gt"]
        )
        ** 2
    ).abs()
    attribution_df["squared_difference"] = attribution_df["abs_difference"] ** 2
    attribution_df = attribution_df.reset_index()
    attribution_df = rank_based_on_column_per_query(
        attribution_df,
        name_column_to_rank="attribution_value_exp",
        new_column_name="exp_ranked",
        biggest_first=True,
    )
    attribution_df = rank_based_on_column_per_query(
        attribution_df,
        name_column_to_rank="attribution_value_gt",
        new_column_name="gt_ranked",
        biggest_first=True,
    )
    attribution_df["abs_rank_difference"] = (
        attribution_df["exp_ranked"] - attribution_df["gt_ranked"]
    ).abs()

    def metrics_on_subset(sub_df, r_df, suffix_metric=""):
        r_df["L1_norm" + suffix_metric] = sub_df.groupby(["query_number"])[
            "abs_difference"
        ].mean()
        r_df["L2_norm" + suffix_metric] = sub_df.groupby(["query_number"])[
            "squared_difference"
        ].mean()
        r_df["spearmans_footrule_metric" + suffix_metric] = sub_df.groupby(
            ["query_number"]
        )["abs_rank_difference"].mean()
        return r_df

    results_df = metrics_on_subset(attribution_df, results_df, suffix_metric="")

    # top-k_metrics
    for i in [1, 3, 5, 10]:
        top_k = attribution_df[attribution_df.gt_ranked <= i]
        results_df = metrics_on_subset(top_k, results_df, suffix_metric="@" + str(i))

    return results_df
