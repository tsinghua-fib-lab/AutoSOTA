from .template import Template
from .event import Event


class EventTemplate(Template):
    def __init__(self, pattern):
        self.pattern = pattern

        self.keyset = set()
        self.logkeyset = set()

        self._find_keys(self.pattern)
        
    def __repr__(self):
        pass

    def _find_keys(self, pattern, header=None):
        # recursively find all keys in the pattern, and keep the structure, e.g. {"a": {"b": ""}} -> {"a", "a.b"}
        if header is None:
            header = ""
        if isinstance(pattern, dict):
            for key, value in pattern.items():
                if isinstance(value, dict):
                    self.keyset.add(f"{header}{key}")
                    self._find_keys(value, f"{header}{key}.")
                else:
                    self.keyset.add(f"{header}{key}")
                    if value != "":
                        self.logkeyset.add(f"{header}{key}")
        
    def is_match(self, e: dict):
        e = Event(data=e)

        print(f"{self.keyset=} {e.keyset=}")
        # check if the event matches the pattern
        if e.keyset != self.keyset:
            return False
        return True
        
    def load_templates(self, template_file):
        pass


