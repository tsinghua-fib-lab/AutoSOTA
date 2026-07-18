from openai import OpenAI
from google import genai
from scenarios.donor.runner import DonorGameRunner, DonorGameRunnerWithGreedyAgent
from scenarios.pd.runner import PDRunner, PDRunnerrWithGreedyAgent
from scenarios.trust.runner import TrustGameRunner
from scenarios.market.runner import ProductChoiceMarketRunner
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from datetime import datetime
from together import Together

# os.environ["WANDB_MODE"] = "disabled" # Set to "disabled" if you don't want to log to wandb, "offline" for local logging
# is_test = True  # Set True for testing
is_test = False

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    root_dir = cfg.metadata.save_dir
    if is_test:
        root_dir = f"{root_dir}TEST/"
    directory = f"{root_dir}logs/{cfg.experiment.env.game_name}_horizon_{cfg.experiment.env.horizon}_gossip_{cfg.experiment.agents.is_gossip}_greedy_{cfg.experiment.agents.insert_greedy_agent}/"
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.metadata.trial_timestamp = timestamp
    log_path = f"{directory}/{timestamp}.json"

    if cfg.llm.api == 'openai':
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif cfg.llm.api == 'gemini':
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    elif cfg.llm.api == 'gemini-v2':
        client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta")
    elif cfg.llm.api == 'deepseek':
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    elif cfg.llm.api == 'together':
        client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    else:
        raise ValueError("Invalid API. Choose 'openai' or 'deepseek'.")

    if cfg.experiment.env.game_name == 'donor':
        if cfg.experiment.agents.insert_greedy_agent:
            runner = DonorGameRunnerWithGreedyAgent(cfg, client, log_path)
        else:
            runner = DonorGameRunner(cfg, client, log_path)
        runner.run_simulation(is_test)
    elif cfg.experiment.env.game_name == 'pd':
        if cfg.experiment.agents.insert_greedy_agent:
            runner = PDRunnerrWithGreedyAgent(cfg, client, log_path)
        else:
            runner = PDRunner(cfg, client, log_path)
        runner.run_simulation(is_test)
    elif cfg.experiment.env.game_name == 'trust':
        runner = TrustGameRunner(cfg, client, log_path)
        runner.run_simulation(is_test)
    elif cfg.experiment.env.game_name == 'market':
        runner = ProductChoiceMarketRunner(cfg, client, log_path)
        runner.run_simulation(is_test)
    else:
        raise ValueError(f"Invalid game. Choose 'donor', 'pd', or 'trust'.")

if __name__ == "__main__":
    main()