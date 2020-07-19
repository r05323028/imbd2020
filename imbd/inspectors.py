import plotly.express as px


class ResultInspector:
    def __init__(self, train_features, train_labels, predictions):
        self.train_features = train_features
        self.train_labels = train_labels
        self.predictions = predictions