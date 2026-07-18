
from pydantic import BaseModel

class SellerActionResponse(BaseModel):
    justification: str
    seller_action: str

class BuyerActionResponse(BaseModel):
    justification: str
    buyer_action: str

class BuyerGossipResponse(BaseModel):
    justification: str
    tone: str
    gossip: str