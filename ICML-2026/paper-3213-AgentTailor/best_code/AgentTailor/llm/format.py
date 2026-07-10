import dataclasses
from typing import List, Literal

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message:
    def __init__(self,role: MessageRole,content: str):
        self.role=role
        self.content=content
    def dict(self):
        return {"role":self.role,"content":self.content}


@dataclasses.dataclass
class Status:
    started: int = 0
    in_progress: int = 0
    succeeded: int = 0
    failed: int = 0
