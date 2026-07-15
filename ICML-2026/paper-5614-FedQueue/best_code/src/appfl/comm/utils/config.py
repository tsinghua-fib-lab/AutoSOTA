from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ClientTask:
    task_id: str = ""
    task_name: str = ""
    client_id: str = ""
    pending: bool = True
    success: bool = False
    start_time: float = -1
    end_time: float = -1
    log: Optional[Dict] = field(default_factory=dict)
