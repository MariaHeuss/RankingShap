import lime
import numpy as np
from utils.helper_functions import rank_list
from utils.explanation import AttributionExplanation


class AggregatedLime:
    def __init__(
        self, background_data, model, explanation_size=3, name="", aggregate_over_top=5
    ):
        self.background_data = background_data
        self.explanation_size = explanation_size
        self.name = name
        self.num_features = len(background_data[0])
        self.model = model
        self.aggregate_over_top = aggregate_over_top

    def get_query_explanation(self, query_features, query_id=""):
        feature_amount = len(query_features[0])

        # create lime explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.background_data,
            feature_names=[i for i in range(0, feature_amount)],
            mode="regression",
        )

        # pick the top ranked document for aggregation of feature importance
        pred = self.model(query_features)
        og_rank = rank_list(pred)

        # Get aggregated importance of each feature for first few (aggregate_over_top) documents
        exp_dict = {i: 0 for i in range(1, self.num_features + 1)}
        for j in range(1, min(self.aggregate_over_top + 1, len(og_rank))):
            best_doc_ind = np.where(og_rank == j)[0][0]
            exp = lime_explainer.explain_instance(
                query_features[best_doc_ind],
                self.model,
                num_features=feature_amount,
            )
            exp_dict = {
                feature: exp_dict.get(feature, 0) + value
                for feature, value in exp.local_exp[0]
            }

        exp_dict = {feature: abs(exp_dict[feature]) for feature in exp_dict}
        # exp_dict = {feature: exp_dict[feature]/sum(exp_dict.values()) for feature in exp_dict}
        exp_dict = {
            feature + 1: exp_dict[feature] for feature in exp_dict
        }  # We start counting features at 1

        explanation = list(exp_dict.items())
        feature_attributes = AttributionExplanation(
            explanation=explanation, num_features=self.num_features, query_id=query_id
        )
        feature_selection = feature_attributes.get_top_k_feature_selection(
            self.explanation_size
        )
        return feature_selection, feature_attributes
