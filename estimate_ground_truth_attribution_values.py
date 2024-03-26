from utils.get_explanations import calculate_all_query_explanations
from utils.helper_functions import get_data
import lightgbm
from utils.background_data import BackgroundData
from approaches.ranking_shap import RankingShap
from scipy.stats import kendalltau
import pandas as pd
from pathlib import Path


import argparse

parser = argparse.ArgumentParser(description="Your script description")

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
    "--background_samples",
    required=True,
    type=int,
    help="Number of background samples to use",
)
parser.add_argument(
    "--nsamples",
    required=True,
    type=int,
    help="Number background, permutation pairs to use (log2)",
)
parser.add_argument(
    "--experiment_iterations",
    required=True,
    type=int,
    help="Run the experiment several times to determine stability of the results",
)
parser.add_argument("--test", action="store_true", help="If true runs only one query")


args = parser.parse_args()
print(args, flush=True)

dataset = args.dataset
experiment_iterations = args.experiment_iterations
n_samples = 2**args.nsamples
background_samples = args.background_samples

explanation_size = 5

if args.test:
    num_queries_eval = 1
else:
    num_queries_eval = None

progress = False

# Get train, test_data
data_directory = Path("data/" + dataset + "/Fold1/")
train_data = get_data(data_file=data_directory / "train.txt")
test_data = get_data(data_file=data_directory / "test.txt")

num_features = len(test_data[0][0])

rank_similarity_coefficient = lambda x, y: kendalltau(x, y)[0]


# We assume that the model has been trained and saved in a model file
model_file = str(args.model_file)
model_directory = Path("results/model_files/")
print(model_directory / model_file)
model = lightgbm.Booster(model_file=str((model_directory / model_file).absolute()))

path_to_attribution_folder = Path("results/results_" + dataset + "/feature_attributes/")


attributes = []

for i in range(experiment_iterations):
    background_data = BackgroundData(
        train_data[0],
        summarization_type="random_sample",
        summary_length=background_samples,
    )

    ground_truth_explainer = RankingShap(
        permutation_sampler="sampling",
        background_data=background_data.background_summary,
        original_model=model.predict,
        explanation_size=explanation_size,
        name="feature_importance_approxiation_"
        + "_backgroundcount_"
        + str(background_samples)
        + "_nsamples_"
        + str(n_samples),
        nsample_permutations=n_samples,
        rank_similarity_coefficient=rank_similarity_coefficient,
    )

    if args.test:
        path_to_attribute_values = path_to_attribution_folder / (
            ground_truth_explainer.name + "_" + str(i) + "_test.csv"
        )
    else:
        path_to_attribute_values = path_to_attribution_folder / (
            ground_truth_explainer.name + "_" + str(i) + ".csv"
        )

    print("Starting iteration ", i, flush=True)
    print("Target csv will be ", path_to_attribute_values, flush=True)

    calculate_all_query_explanations(
        explainer=ground_truth_explainer,
        eval_data=test_data,
        num_queries_to_eval=num_queries_eval,
        progress=True,
        safe_attributions_to=path_to_attribute_values,
    )

    experiment_results = pd.read_csv(path_to_attribute_values)
    experiment_results = experiment_results.set_index("feature_number")
    experiment_results = experiment_results.stack().swaplevel().sort_index()
    experiment_results = experiment_results.reset_index().rename(
        columns={"level_0": "query_number", 0: "attribution_value"}
    )

    attributes.append(experiment_results)

attributes = pd.concat(attributes)

means = attributes.groupby(["query_number", "feature_number"])[
    "attribution_value"
].mean()
stds = attributes.groupby(["query_number", "feature_number"])["attribution_value"].std()
ground_truth_attributes = pd.DataFrame({"attribution_value": means, "std": stds})

if args.test:
    path_to_attribute_values = path_to_attribution_folder / (
        ground_truth_explainer.name + "_means_test.csv"
    )
else:
    path_to_attribute_values = path_to_attribution_folder / (
        ground_truth_explainer.name + "_means.csv"
    )

ground_truth_attributes.reset_index().to_csv(path_to_attribute_values)
