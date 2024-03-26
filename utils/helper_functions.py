import numpy as np

# helper function, rank list with highest rank for highest value
import pandas as pd
from pyltr.data.letor import read_dataset


def test_rank_list():
    scores1 = np.array([0, 1, 2, 3])
    rank1 = rank_list(scores1)
    scores2 = np.array([1, 0, 3, 2])
    rank2 = rank_list(scores2)
    scores3 = np.array([2, 1, 0, 3])
    rank3 = rank_list(scores3)
    assert np.all(rank1 == np.array([4, 3, 2, 1]))
    assert np.all(rank2 == np.array([3, 4, 1, 2]))
    assert np.all(rank3 == np.array([2, 3, 4, 1]))


def rank_list(vector):
    """
    returns ndarray containing rank(i) for documents at position i
    """
    temp = vector.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(vector) + 1)

    return ranks


def rank_based_on_column_per_query(
    data_frame,
    name_column_to_rank,
    new_column_name,
    name_query_column="query_number",
    biggest_first=True,
):
    dfs = []
    for query in set(data_frame[name_query_column].values):
        data_frame_query = data_frame[data_frame[name_query_column] == query]
        data_frame_query = data_frame_query.sample(frac=1).reset_index(
            drop=True
        )  # Shuffle rows for breaking ties randomly
        data_frame_query[new_column_name] = (
            data_frame_query[[name_column_to_rank]]
            .apply(tuple, axis=1)
            .rank(method="first", ascending=False)
            .astype(int)
        )
        dfs.append(data_frame_query)
    return pd.concat(dfs, axis=0)


def test_rank_based_on_column_per_query():
    data_frame = pd.DataFrame(
        {"query_number": [1, 1, 1, 2, 2, 2], "attribution_value": [3, 2, 1, 1, 2, 3]}
    )
    data_frame = rank_based_on_column_per_query(
        data_frame,
        name_column_to_rank="attribution_value",
        new_column_name="ranked",
        biggest_first=True,
    )
    assert list(data_frame.ranked.values) == [1, 2, 3, 3, 2, 1]


def get_data(data_file):
    with open(data_file) as evalfile:
        EX, Ey, Eqids, _ = read_dataset(evalfile)
    return EX, Ey, Eqids


def get_queryids_as_list(Eqids):
    qids = []
    [qids.append(x) for x in Eqids if x not in qids]
    return qids


def get_documents_per_query(Eqids):
    # determine the amount of queries in the to be analysed set and construct a countlist
    df = pd.DataFrame(data=Eqids, columns=["Eqids"])
    qid_count_list = df.groupby("Eqids")["Eqids"].count()
    return qid_count_list


if __name__ == "__main__":
    test_rank_list()
    test_rank_based_on_column_per_query()
