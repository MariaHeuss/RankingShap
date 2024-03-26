import warnings
import pandas as pd
from utils.helper_functions import get_queryids_as_list, get_documents_per_query
import os
from pathlib import Path

warnings.filterwarnings("ignore")


def calculate_all_query_explanations(
    explainer,
    eval_data,
    num_queries_to_eval=None,
    progress=False,
    safe_attributions_to=None,
):
    EX, _, Eqids = eval_data
    queries = get_queryids_as_list(Eqids)

    if num_queries_to_eval is None:
        queries_to_eval = queries
        print("Evaluating all ", len(queries), " queries")

    else:
        queries_to_eval = queries[:num_queries_to_eval]
        print("Evaluating the first", len(queries_to_eval), " queries")

    # keep track of query start position in test set
    list_trckr = 0
    i = 1
    qid_count_list = get_documents_per_query(Eqids)

    if safe_attributions_to is not None and os.path.isfile(safe_attributions_to):
        os.remove(safe_attributions_to)

    # select all query document pairs for a certain query and calculate the validity and completeness
    for query in queries:
        query_len = qid_count_list[query]
        if query in queries_to_eval:
            # define the current query to pass to the greedy algorithm to find explanation
            current_query = EX[list_trckr : (list_trckr + query_len)]

            features_selection, feature_attribution = explainer.get_query_explanation(
                query_features=current_query, query_id=query
            )

            feature_attribution.safe_to_file(safe_attributions_to)

            if progress:
                print(str(i) + "/" + str(len(queries_to_eval)))
            i += 1
        # update list tracker
        list_trckr += query_len

    def prepare_for_eval(path_to_attribute_values):
        experiment_results = pd.read_csv(path_to_attribute_values)
        experiment_results = experiment_results.set_index("feature_number")
        experiment_results = experiment_results.stack().swaplevel().sort_index()
        experiment_results = experiment_results.reset_index().rename(
            columns={"level_0": "query_number", 0: "attribution_value"}
        )
        experiment_results = experiment_results.set_index(
            ["query_number", "feature_number"]
        )
        experiment_results.to_csv(
            Path(str(path_to_attribute_values).split(".")[0] + "_eval.csv")
        )

    prepare_for_eval(safe_attributions_to)
    return
