import lightgbm
import pandas as pd
from utils.helper_functions import get_data
from utils.background_data import BackgroundData
import numpy as np
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(description="Your script description")

parser.add_argument(
    "--file_name",
    required=True,
    type=str,
    help="Path to the file where we want to store the model parameters",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["MSLR-WEB10K", "MQ2008"],
    help="The dataset to use MQ2008 or MSLR-WEB10K",
)
parser.add_argument(
    "--model_type",
    required=True,
    type=str,
    choices=["listwise", "pairwise" or "pointwise"],
    help="Type of model we want to use, either listwise, pairwise or pointwise",
)
parser.add_argument(
    "--background_samples",
    required=False,
    type=int,
    default=100,
    help="Number of background samples to be used for explanations",
)

args = parser.parse_args()
print(args, flush=True)

dataset = args.dataset
file_name = args.file_name
model_type = args.model_type
background_samples = args.background_samples


def train_model(
    train_data, eval_data, ranking_model_type="listwise", save_to_file="model.txt"
):
    assert ranking_model_type in ["listwise", "pairwise", "pointwise"]

    TX, Ty, Tqids = train_data
    VX, Vy, Vqids = eval_data

    # create pandas dataframe of qids to easily construct group list
    T_df = pd.DataFrame(data=Tqids, columns=["Tqids"])
    V_df = pd.DataFrame(data=Vqids, columns=["Vqids"])
    qid_count_list_train = T_df.groupby("Tqids")["Tqids"].count().to_numpy()
    qid_count_list_val = V_df.groupby("Vqids")["Vqids"].count().to_numpy()

    # model training
    if ranking_model_type == "listwise":
        # train lambda rank model
        model = lightgbm.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
        )

        model.fit(
            X=TX,
            y=Ty,
            group=qid_count_list_train,
            eval_set=[(VX, Vy)],
            eval_group=[qid_count_list_val],
            eval_at=10,
        )

    if ranking_model_type == "pairwise":
        # train lambda rank model
        model = lightgbm.LGBMRanker(
            objective="binary",
            metric="ndcg",
        )

        model.fit(
            X=TX,
            y=Ty,
            group=qid_count_list_train,
            eval_set=[(VX, Vy)],
            eval_group=[qid_count_list_val],
            eval_at=10,
        )

    if ranking_model_type == "pointwise":
        # train lambda rank model
        model = lightgbm.LGBMRanker(
            objective="regression",
            metric="ndcg",
        )

        model.fit(
            X=TX,
            y=Ty,
            group=qid_count_list_train,
            eval_set=[(VX, Vy)],
            eval_group=[qid_count_list_val],
            eval_at=10,
        )
    model.booster_.save_model(save_to_file)
    return model


# Get train, eval_data
data_directory = Path("data/" + dataset + "/Fold1/")
train_data = get_data(data_file=data_directory / "train.txt")
eval_data = get_data(data_file=data_directory / "vali.txt")

model = train_model(
    train_data=train_data,
    eval_data=eval_data,
    ranking_model_type=model_type,
    save_to_file=Path("results/model_files/" + file_name),
)

# Prepare background data training:
background_data = BackgroundData(
    train_data[0], summarization_type="random_sample", summary_length=background_samples
)
np.save(
    Path("results/background_data_files/train_background_data_" + dataset + ".npy"),
    background_data.background_summary,
)
