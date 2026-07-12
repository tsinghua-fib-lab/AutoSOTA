from .feedback import Feedback

class PreferenceFeedback(Feedback):
    def __init__(self, prec_id: int, succ_id: int) -> None:
        if prec_id == succ_id:
            raise ValueError(f"Strict inequality violation: Item cannot precede itself (id: {prec_id})")
        
        self.prec_id = prec_id
        self.succ_id = succ_id
