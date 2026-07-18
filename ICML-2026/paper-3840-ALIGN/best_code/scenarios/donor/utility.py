from pydantic import BaseModel
import numpy as np

class BinaryDonationResponse(BaseModel):
    justification: str
    donor_action: str

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

def compute_avg_donation(agent):
    """
    Calculate the average donation made by an agent.
    """
    return 0 if len(agent.donations) == 0 else np.mean(agent.donations)

def compute_donation_ratio(donation, total_resources):
    """
    Calculate the average donation ratio for an agent.
    The ratio is computed as the average donation divided by the initial resources.
    """
    return donation / total_resources if total_resources > 0 else 0

def compute_avg_donation_ratio(agent):
    return 0 if len(agent.donation_ratios) == 0 else np.mean(agent.donation_ratios)

def compute_image_score(agent):
    """
    Calculate the image score for an agent based on their donations.
    """
    image_score = 0
    for donation in agent.donations:
        if donation > 0:
            image_score += 1
        else:
            image_score -= 1
    return image_score