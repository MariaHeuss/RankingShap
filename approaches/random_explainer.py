import numpy as np
from utils.explanation import AttributionExplanation


class RandomExplainer:
    def __init__(self, num_features, explanation_size=3, name="random"):
        self.explanation_size = explanation_size
        self.name = name
        self.num_features = num_features

    def get_query_explanation(self, query_features, query_id=""):
        explanation = np.random.random(self.num_features)
        explanation = list(explanation / sum(explanation))
        explanation = [(i + 1, explanation[i]) for i in range(self.num_features)]

        feature_attributes = AttributionExplanation(
            explanation=explanation, num_features=self.num_features, query_id=query_id
        )
        feature_selection = feature_attributes.get_top_k_feature_selection(
            self.explanation_size
        )
        return feature_selection, feature_attributes
