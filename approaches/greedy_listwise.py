import numpy as np
from evaluation.evaluate_explanations import calculate_validity_completeness
from utils.explanation import SelectionExplanation, AttributionExplanation


class GreedyListwise:
    def __init__(
        self,
        background_data,
        model,
        explanation_size=3,
        name="",
        feature_attribution_method="iter",
        mixed_type_input=False,
    ):
        self.background_data = background_data
        self.explanation_size = explanation_size
        self.name = name
        self.feature_shape = np.shape(background_data[0])
        self.num_features = len(background_data[0])
        self.model = model
        self.original_model = model
        self.feature_attribution_method = feature_attribution_method
        self.mixed_type_input = mixed_type_input

    def get_query_explanation(self, query_features, query_id=""):
        feature_set = np.arange(len(query_features[0]))
        explanation = []

        for it in range(self.explanation_size):
            max_score = -np.inf
            best_f = 0

            # to find the next best feature iterate over all features f that are not yet used in the explanation
            for f in feature_set:
                # add f to the explanation consisting of already selected features
                used_fs = np.append(f, np.array(explanation.copy(), dtype="int"))

                val_score, comp_score = calculate_validity_completeness(
                    query_features,
                    self.model,
                    used_fs,
                    background_data=self.background_data,
                    mixed_type_input=self.mixed_type_input,
                )

                # we chose f with the maximal score:
                if val_score > max_score:
                    max_score = val_score
                    best_f = f
            explanation.append(best_f)
            feature_set = np.delete(feature_set, np.where(feature_set == best_f))

        feature_selection = SelectionExplanation(
            explanation, num_features=self.num_features, query_id=query_id
        )

        feature_attr = self.get_feature_attribution(
            explanation,
            query_features,
            feature_attribution_method=self.feature_attribution_method,
            query_id=query_id,
        )
        return feature_selection, feature_attr

    def get_feature_attribution(
        self,
        feature_selection_explanation,
        query_features,
        feature_attribution_method="naive",
        query_id="",
    ):
        assert feature_attribution_method in ["iter", "marginal"]
        self.feature_attribution_method = feature_attribution_method

        feature_attribution_explanation = [(i, 0) for i in range(self.num_features)]

        if feature_attribution_method == "iter":
            used_features = []
            previous_score = 0
            for feature in feature_selection_explanation:
                used_features.append(feature)
                val_score, comp_score = calculate_validity_completeness(
                    query_features,
                    self.model,
                    used_features,
                    background_data=self.background_data,
                    mixed_type_input=self.mixed_type_input,
                )

                feature_attribution_explanation[feature] = (
                    feature,
                    val_score - previous_score,
                )
                previous_score = val_score

        elif feature_attribution_method == "marginal":
            val_score, comp_score = calculate_validity_completeness(
                query_features,
                self.model,
                feature_selection_explanation,
                background_data=self.background_data,
                mixed_type_input=self.mixed_type_input,
            )

            for feature in feature_selection_explanation:
                f = feature_selection_explanation.copy()
                f.remove(feature)
                val_score_wo_feature, comp_score_wo_feature = (
                    calculate_validity_completeness(
                        query_features,
                        self.model,
                        f,
                        background_data=self.background_data,
                        mixed_type_input=self.mixed_type_input,
                    )
                )
                feature_attribution_explanation[feature] = (
                    feature,
                    val_score - val_score_wo_feature,
                )
        else:
            raise NotImplementedError("Unsupported attribution method")
        # Start counting features at 1:
        feature_attribution_explanation = [
            (feature + 1, value) for (feature, value) in feature_attribution_explanation
        ]
        return AttributionExplanation(
            explanation=feature_attribution_explanation,
            num_features=self.num_features,
            query_id=query_id,
        )
