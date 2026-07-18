from pydantic import BaseModel
import numpy as np

class ActionResponse(BaseModel):
    justification: str
    player_action: str

class GossipResponse(BaseModel):
    justification: str
    tone: str
    gossip: str

def compute_return(agent, resources_start): 
    return agent.resources - resources_start

def compute_dis_cum_reward(agent, discount_factor):
    """
    Calculate the discounted cumulative reward for an agent.
    """
    discounted_cumulative_reward = 0
    for idx, reward in enumerate(agent.rewards):
        discounted_cumulative_reward += (discount_factor ** idx) * reward
    return discounted_cumulative_reward

def compute_image_score(agent):
    """
    Calculate the image score for an agent based on their donations.
    """
    image_score = 0
    for action in agent.actions:
        if action == "C":
            image_score += 1
        else:
            image_score -= 1
    return image_score

def compute_cooperation_ratio(agent):
    """
    Calculate the cooperation ratio for an agent.
    """
    if len(agent.actions) == 0:
        return 0.0
    cooperation_count = sum(1 for action in agent.actions if action == "C")
    return cooperation_count / len(agent.actions)