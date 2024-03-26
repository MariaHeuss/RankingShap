import numpy as np
import pandas as pd
from utils.helper_functions import rank_list

from scipy.stats import kendalltau


def calculate_validity_completeness(
    query_features,
    model,
    explanation,
    background_data,
    rank_similarity_coefficient=lambda x, y: kendalltau(x, y)[0],
    mixed_type_input=False,
):
    feature_set = np.arange(len(query_features[0]))
    num_features = len(feature_set)
    # Determine mask templates, both for validity and completeness
    mask = np.array([i in explanation for i in range(num_features)])

    if mixed_type_input:
        # For mixed type arrays we need to use pandas.where but for efficiency we want to use np.array...
        # Use (the same) mask on feature vectors of each document
        val_vectors = np.array(
            [
                [
                    pd.Series(doc_features).where(mask, background_vector)
                    for doc_features in query_features
                ]
                for background_vector in background_data
            ]
        )
        comp_vectors = np.array(
            [
                [
                    pd.Series(background_vector).where(mask, doc_features)
                    for doc_features in query_features
                ]
                for background_vector in background_data
            ]
        )

    else:
        # Use (the same) mask on feature vectors of each document
        val_vectors = np.array(
            [
                [
                    np.where(mask, doc_features, background_vector)
                    for doc_features in query_features
                ]
                for background_vector in background_data
            ]
        )
        comp_vectors = np.array(
            [
                [
                    np.where(mask, background_vector, doc_features)
                    for doc_features in query_features
                ]
                for background_vector in background_data
            ]
        )

    # Predict ranking scores with and without applied masks
    pred = model(query_features)
    val_preds = [model(vectors) for vectors in val_vectors]
    comp_preds = [model(vectors) for vectors in comp_vectors]

    val_scores = [
        rank_similarity_coefficient(rank_list(val_pred), rank_list(pred))
        for val_pred in val_preds
    ]
    comp_scores = [
        -rank_similarity_coefficient(rank_list(comp_pred), rank_list(pred))
        for comp_pred in comp_preds
    ]
    return np.mean(val_scores), np.mean(comp_scores)
