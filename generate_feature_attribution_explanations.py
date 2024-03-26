from utils.get_explanations import calculate_all_query_explanations
from utils.helper_functions import get_data
import lightgbm
import numpy as np
from scipy.stats import kendalltau
from utils.BackgroundData import BackgroundData
from approaches.RankingSHAP import RankingShap
from approaches.RankingLime import RankingLIME
from approaches.GreedyListwise import GreedyListwise
from approaches.PointwiseLime import AggregatedLime
from approaches.PointwiseSHAP import AggregatedShap
from approaches.RandomExplainer import RandomExplainer

import argparse

parser = argparse.ArgumentParser(description="Runs different explanation approaches")

parser.add_argument(
    "--model_file",
    required=True,
    type=str,
    help="Path to the model file of the model that we want to approximate the feature importance for",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["MSLR-WEB10K", "MQ2008"],
    help="The dataset to use MQ2008 or MSLR-WEB10K",
)
parser.add_argument(
    "--experiment_iteration",
    required=True,
    type=int,
    help="Enables us to run the same experiment several times",
)
parser.add_argument("--test", action="store_true", help="If true runs only one query")


args = parser.parse_args()
print(args, flush=True)

dataset = args.dataset
experiment_iteration = args.experiment_iteration
test = args.test

# We assume that the model has been trained and saved in a model file
model_file = args.model_file
#
# dataset = "MQ2008"
# experiment_iteration = 1
# n_samples = 2 ** 10
# test = True
# model_file = 'test'

model = lightgbm.Booster(model_file="results/model_files/" + model_file)


explanation_size = 5


progress = False
if test:
    num_queries_eval = 1
else:
    num_queries_eval = None

# Get train, eval_data
train_data = get_data(path_to_data="data/" + dataset + "/Fold1/train.txt")
test_data = get_data(path_to_data="data/" + dataset + "/Fold1/test.txt")
eval_data = get_data(path_to_data="data/" + dataset + "/Fold1/vali.txt")

path_to_estimated_feature_importance = (
    "results/results_"
    + dataset
    + "/feature_attributes/"
    + "attribution_values_for_evaluation.csv"
)

num_features = len(test_data[0][0])

background_data = BackgroundData(
    np.load("results/background_data_files/train_background_data_" + dataset + ".npy"),
    summarization_type=None,
)

rank_similarity_coefficient = lambda x, y: kendalltau(x, y)[0]

# Define all the explainers
ranking_shap_explainer = RankingShap(
    permutation_sampler="kernel",
    background_data=background_data.background_summary,
    original_model=model,
    explanation_size=explanation_size,
    name="rankingshap",
    rank_similarity_coefficient=rank_similarity_coefficient,
)


greedy_explainer_0_iter = GreedyListwise(
    background_data=background_data.background_summary,
    model=model,
    explanation_size=explanation_size,
    name="greedy_iter",
    feature_attribution_method="iter",
)


greedy_explainer_0_full = GreedyListwise(
    background_data=background_data.background_summary,
    model=model,
    explanation_size=num_features,
    name="greedy_iter_full",
    feature_attribution_method="iter",
)


aggregated_lime_explainer = AggregatedLime(
    background_data=background_data.background_summary,
    model=model,
    explanation_size=explanation_size,
    name="pointwise_lime",
    aggregate_over_top=5,
)

aggregated_shap_explainer = AggregatedShap(
    background_data=background_data.background_summary,
    model=model,
    explanation_size=explanation_size,
    name="pointwise_shap",
    aggregate_over_top=5,
    nsamples=2**10,
)

random_explainer = RandomExplainer(
    explanation_size=explanation_size, name="random", num_features=num_features
)

ranking_lime_explainer = RankingLIME(
    background_data=background_data.background_summary,
    original_model=model,
    explanation_size=explanation_size,
    name="rankinglime",
    rank_similarity_coefficient=rank_similarity_coefficient,
    use_entry=0,
    individual_masking=True,
)

explainers = [
    random_explainer,
    aggregated_shap_explainer,
    aggregated_lime_explainer,
    ranking_shap_explainer,
    greedy_explainer_0_iter,
    ranking_lime_explainer,
]

if dataset == "MQ2008":
    explainers.append(greedy_explainer_0_full)

for exp in explainers:
    if test:
        path_to_attribute_values = (
            "results/results_"
            + dataset
            + "/feature_attributes/"
            + exp.name
            + "_test.csv"
        )
    else:
        path_to_attribute_values = (
            "results/results_" + dataset + "/feature_attributes/" + exp.name + ".csv"
        )

    print("Starting", exp.name, flush=True)
    print("Target csv will be ", path_to_attribute_values, flush=True)

    calculate_all_query_explanations(
        explainer=exp,
        eval_data=test_data,
        num_queries_to_eval=num_queries_eval,
        progress=True,
        safe_attributions_to=path_to_attribute_values,
    )
