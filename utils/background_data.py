import shap


class BackgroundData:
    def __init__(self, data, summarization_type=None, summary_length=1):
        self.data = data
        self.summarization_type = summarization_type
        self.summary_length = summary_length
        self.background_summary = self.get_background_summary()

    def get_background_summary(self):
        if self.summarization_type is None:
            background_summary = self.data
        elif self.summarization_type == "random_sample":
            background_summary = shap.sample(self.data, self.summary_length)
        else:
            NotImplementedError
        return background_summary
