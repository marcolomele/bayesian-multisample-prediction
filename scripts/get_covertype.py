class CovertypeDataset:
    def __init__(self):
        from ucimlrepo import fetch_ucirepo
        self.dataset = fetch_ucirepo(id=31)
        self.X = self.dataset.data.features
        self.y = self.dataset.data.targets
        self.metadata = self.dataset.metadata
        self.variables = self.dataset.variables

    def get_features(self):
        return self.X

    def get_targets(self):
        return self.y

    def get_metadata(self):
        return self.metadata

    def get_variables(self):
        return self.variables
