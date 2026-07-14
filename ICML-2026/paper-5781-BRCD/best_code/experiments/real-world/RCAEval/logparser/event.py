class Event:
    def __init__(self, data):
        self.data = data
        self.keyset = set()
        self.properties = dict()
        self._find_keys(data)

    def _find_keys(self, data, header=None):
        assert isinstance(data, dict), "event data must be a dict"
        if header is None:
            header = ""
        if isinstance(data, dict):
            for key, value in data.items():
                self.keyset.add(f"{header}{key}")
                self.properties[f"{header}{key}"] = value
                if isinstance(value, dict):
                    self._find_keys(value, f"{header}{key}.")
