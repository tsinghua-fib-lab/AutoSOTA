import os

# Read OpenAI API Key from environment variable
# export OPENAI_API_KEY="your-key-here"
openai_api_key = os.getenv("OPENAI_API_KEY", "")

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

# Verbose
debug = True
