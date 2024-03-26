import pandas as pd
from evaluation.evaluate_feature_attribution import eval_feature_attribution
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description="Your script description")

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["MSLR-WEB10K", "MQ2008"],
    help="The dataset to use MQ2008 or MSLR-WEB10K",
)
parser.add_argument(
    "--file_name_ground_truth",
    required=True,
    type=str,
    help="File name of truth attribution values",
)
parser.add_argument(
    "--test", action="store_true", help="If true uses test files for evaluation"
)


args = parser.parse_args()
print(args, flush=True)

dataset = args.dataset
file_name_ground_truth = args.file_name_ground_truth
path_to_attribution_folder = Path("results/results_" + dataset + "/feature_attributes/")

path_to_ground_truth_attributes = path_to_attribution_folder / file_name_ground_truth


approaches = [
    "rankingshap",
    "greedy_iter",
    "greedy_iter_full",
    "pointwise_lime",
    "pointwise_shap",
    "random",
    "random",
    "rankinglime",
]
if args.test:
    approaches = [a + "_test" for a in approaches]

eval_df = []


for approach in approaches:
    path_to_attribute_values = path_to_attribution_folder / (
        approach + "_eval" + ".csv"
    )

    attribution_evaluation_per_query = eval_feature_attribution(
        attributes_to_evaluate=path_to_attribute_values,
        ground_truth_file_path=path_to_ground_truth_attributes,
    )

    mean_attribute_evaluation = attribution_evaluation_per_query.mean()
    mean_attribute_evaluation["approach"] = approach
    eval_df.append(mean_attribute_evaluation)


mean_attribute_evaluation = pd.DataFrame(eval_df)

mean_attribute_evaluation = mean_attribute_evaluation.set_index(["approach"])

evaluation_for_table = mean_attribute_evaluation[
    [
        "spearmans_footrule_metric",
        "spearmans_footrule_metric@3",
        "spearmans_footrule_metric@10",
        "L1_norm",
        "L1_norm@3",
        "L1_norm@10",
    ]
]
evaluation_for_table = evaluation_for_table.rename(
    {
        "spearmans_footrule_metric": "order",
        "spearmans_footrule_metric@3": "order@3",
        "spearmans_footrule_metric@10": "order@10",
        "L1_norm": "valdis",
        "L1_norm@3": "valdis@3",
        "L1_norm@10": "valdis@10",
    },
    axis=1,
)

evaluation_for_table = evaluation_for_table.round(
    {
        "order": 1,
        "order@3": 1,
        "order@10": 1,
        "valdis": 4,
        "valdis@3": 4,
        "valdis@10": 4,
    }
)

print(evaluation_for_table)
