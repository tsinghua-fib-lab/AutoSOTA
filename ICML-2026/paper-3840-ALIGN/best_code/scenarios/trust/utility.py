from pydantic import BaseModel

class InvestmentResponse(BaseModel):
    justification: str
    investor_action: float

class ReturnResponse(BaseModel):
    justification: str
    responder_action: float

class InvestorGossipResponse(BaseModel):
    justification: str
    tone: str
    gossip: str

class ResponderGossipResponse(BaseModel):
    justification: str
    tone: str
    gossip: str

def extract_json(s: str) -> str:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in string: {repr(s)}")
    return s[start:end+1]