import shap
import numpy as np
from utils.helper_functions import rank_list
from utils.explanation import AttributionExplanation


class AggregatedShap:
    def __init__(
        self,
        background_data,
        model,
        explanation_size=3,
        name="",
        aggregate_over_top=5,
        nsamples=2**10,
    ):
        self.background_data = background_data
        self.explanation_size = explanation_size
        self.name = name
        self.num_features = len(background_data[0])
        self.nsamples = nsamples
        self.model = model
        self.aggregate_over_top = aggregate_over_top

    def get_query_explanation(self, query_features, query_id=""):
        background_summary = shap.sample(
            self.background_data, 100
        )  # create summary of background data

        shap_explainer = shap.KernelExplainer(self.model, background_summary)
        pred = self.model(query_features)

        og_rank = rank_list(pred)

        exp_dict = {i+1: 0 for i in range(self.num_features)}

        aggregate_over_top = min(self.aggregate_over_top + 1, len(og_rank))

        # Get aggregated importance of each feature for first few documents
        for j in range(1, aggregate_over_top):
            best_doc_ind = np.where(og_rank == j)[0][0]

            exp = shap_explainer.shap_values(
                np.array(query_features[best_doc_ind]), nsamples=self.nsamples
            )
            exp_dict = {
                feature + 1: exp_dict.get(feature + 1, 0) + value
                for feature, value in enumerate(exp)
            }

        try:
            # Normalize to 1 to make it better comparable with other methods.
            exp_dict = {
                feature: exp_dict[feature] / sum(exp_dict.values())
                for feature in exp_dict
            }
        except:
            exp_dict = exp_dict
        explanation = list(exp_dict.items())
        feature_attributes = AttributionExplanation(
            explanation=explanation, num_features=self.num_features, query_id=query_id
        )
        feature_selection = feature_attributes.get_top_k_feature_selection(
            self.explanation_size
        )
        return feature_selection, feature_attributes
