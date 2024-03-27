import pandas as pd
import os.path
import random


class Explanation:
    def __init__(self, num_features, query_id=""):
        self.query = query_id
        self.num_features = num_features


class AttributionExplanation(Explanation):
    def __init__(self, explanation, *args, **kwargs):
        assert isinstance(explanation, list)
        self.explanation = explanation
        super().__init__(*args, **kwargs)

    def safe_to_file(self, filename):
        print("Writing to ", filename)
        attribute_dataframe = pd.DataFrame(
            self.explanation, columns=["feature_number", str(self.query)]
        )
        if os.path.isfile(filename):
            attribute_dataframe = attribute_dataframe.merge(
                pd.read_csv(filename), how="outer", on="feature_number"
            )
        attribute_dataframe.to_csv(filename, index=False)

    @classmethod
    def from_file(cls, filename, query_id, num_features):
        a = pd.read_csv(filename)[["feature_number", str(query_id)]]
        explanation = list(a.itertuples(index=False, name=None))
        # Reads the explanation of that specific query and method from a file, where several query explanations could be stored.
        return cls(
            explanation=explanation, query_id=int(query_id), num_features=num_features
        )

    def get_top_k_feature_selection(self, top_k):
        assert top_k <= self.num_features
        exp_dict = sorted(
            self.explanation, key=lambda item: (item[1], random.random()), reverse=True
        )  # For ties we chose a random item.
        selection_explanation = [exp_dict[i][0] for i in range(top_k)]
        return SelectionExplanation(selection_explanation, self.num_features)


class SelectionExplanation(Explanation):
    def __init__(self, explanation, *args, **kwargs):
        self.explanation = explanation
        super().__init__(*args, **kwargs)
